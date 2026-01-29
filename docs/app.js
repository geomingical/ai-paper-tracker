(function() {
    'use strict';

    const CATEGORY_ICONS = {
        'foundations': '🏛️',
        'language-models': '💬',
        'multimodal': '🎨',
        'efficiency': '⚡',
        'data': '📊'
    };

    const CATEGORY_NAMES = {
        'foundations': 'Foundations',
        'language-models': 'Language Models',
        'multimodal': 'Multimodal',
        'efficiency': 'Efficiency',
        'data': 'Data & Scaling'
    };

    let papersData = null;
    let readPapers = new Set();
    let currentFilter = 'all';

    async function init() {
        await loadPapersData();
        await loadReadingNotes();
        renderPapers();
        updateStats();
        updateCategoryCounts();
        setupEventListeners();
    }

    async function loadPapersData() {
        try {
            const response = await fetch('papers.json');
            papersData = await response.json();
        } catch (error) {
            console.error('Failed to load papers data:', error);
            papersData = { categories: [], papers: [] };
        }
    }

    async function loadReadingNotes() {
        readPapers.clear();
        
        for (const paper of papersData.papers) {
            try {
                const response = await fetch(`notes/${paper.id}.md`);
                if (response.ok) {
                    const content = await response.text();
                    if (content.trim().length > 0) {
                        readPapers.add(paper.id);
                        paper.notes = content;
                    }
                }
            } catch (e) {
                // Note file doesn't exist, paper is unread
            }
        }
    }

    function renderPapers() {
        const grid = document.getElementById('papers-grid');
        grid.innerHTML = '';

        const filteredPapers = currentFilter === 'all' 
            ? papersData.papers 
            : papersData.papers.filter(p => p.category === currentFilter);

        const sortedPapers = [...filteredPapers].sort((a, b) => a.year - b.year);

        sortedPapers.forEach((paper, index) => {
            const card = createPaperCard(paper, index);
            grid.appendChild(card);
        });
    }

    function createPaperCard(paper, index) {
        const card = document.createElement('div');
        card.className = `paper-card ${readPapers.has(paper.id) ? 'read' : 'unread'}`;
        card.dataset.category = paper.category;
        card.dataset.paperId = paper.id;
        card.style.animationDelay = `${Math.min(index * 0.05, 0.5)}s`;

        const authorsDisplay = paper.authors.slice(0, 3).join(', ') + 
            (paper.authors.length > 3 ? ` +${paper.authors.length - 3}` : '');

        card.innerHTML = `
            <div class="card-header">
                <span class="card-category">
                    <span class="card-category-icon">${CATEGORY_ICONS[paper.category]}</span>
                    ${CATEGORY_NAMES[paper.category]}
                </span>
                <span class="card-year">${paper.year}</span>
            </div>
            <h3 class="card-title">${paper.title}</h3>
            <p class="card-authors">${authorsDisplay}</p>
            <p class="card-significance">${paper.significance}</p>
            <div class="card-tags">
                ${paper.tags.slice(0, 3).map(tag => `<span class="tag">${tag}</span>`).join('')}
            </div>
        `;

        card.addEventListener('click', () => openModal(paper));
        return card;
    }

    function openModal(paper) {
        const modal = document.getElementById('paper-modal');
        
        document.getElementById('modal-category').innerHTML = 
            `${CATEGORY_ICONS[paper.category]} ${CATEGORY_NAMES[paper.category]}`;
        document.getElementById('modal-year').textContent = paper.year;
        document.getElementById('modal-title').textContent = paper.title;
        document.getElementById('modal-authors').textContent = paper.authors.join(', ');
        document.getElementById('modal-significance').textContent = paper.significance;
        
        const tagsContainer = document.getElementById('modal-tags');
        tagsContainer.innerHTML = paper.tags.map(tag => `<span class="tag">${tag}</span>`).join('');

        const notesContent = document.getElementById('notes-content');
        if (paper.notes && paper.notes.trim()) {
            notesContent.innerHTML = marked ? marked.parse(paper.notes) : paper.notes.replace(/\n/g, '<br>');
            notesContent.classList.add('has-notes');
        } else {
            notesContent.innerHTML = `<p class="no-notes">No reading notes yet. Add a markdown file to <code>notes/${paper.id}.md</code> to track your thoughts!</p>`;
            notesContent.classList.remove('has-notes');
        }

        const arxivBtn = document.getElementById('modal-arxiv');
        if (paper.arxiv) {
            arxivBtn.href = `https://arxiv.org/abs/${paper.arxiv}`;
            arxivBtn.style.display = 'inline-flex';
        } else {
            arxivBtn.style.display = 'none';
        }

        modal.classList.add('open');
        document.body.style.overflow = 'hidden';
    }

    function getCategoryFolder(category) {
        const folderMap = {
            'foundations': '基礎模型',
            'language-models': '語言模型',
            'multimodal': '多模態',
            'efficiency': '模型效率與優化',
            'data': '數據'
        };
        return folderMap[category] || category;
    }

    function closeModal() {
        const modal = document.getElementById('paper-modal');
        modal.classList.remove('open');
        document.body.style.overflow = '';
    }

    function updateStats() {
        const totalCount = papersData.papers.length;
        const readCount = readPapers.size;
        const percentage = totalCount > 0 ? Math.round((readCount / totalCount) * 100) : 0;

        document.getElementById('read-count').textContent = readCount;
        document.getElementById('total-count').textContent = totalCount;
        document.getElementById('progress-percent').textContent = `${percentage}%`;
        
        const progressRing = document.getElementById('progress-ring');
        progressRing.setAttribute('stroke-dasharray', `${percentage}, 100`);
    }

    function updateCategoryCounts() {
        document.getElementById('count-all').textContent = papersData.papers.length;
        
        const categoryCounts = {};
        papersData.papers.forEach(paper => {
            categoryCounts[paper.category] = (categoryCounts[paper.category] || 0) + 1;
        });

        Object.keys(CATEGORY_NAMES).forEach(category => {
            const countEl = document.getElementById(`count-${category}`);
            if (countEl) {
                countEl.textContent = categoryCounts[category] || 0;
            }
        });
    }

    function setupEventListeners() {
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                currentFilter = tab.dataset.category;
                renderPapers();
            });
        });

        document.querySelector('.modal-close').addEventListener('click', closeModal);
        document.querySelector('.modal-backdrop').addEventListener('click', closeModal);
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
