## Plan: Quant Portfolio Repositioning

Reframe the site from a general student portfolio into a quant/scientific case-study hub by restructuring hero messaging, rewriting core projects into PAR format with mathematically rigorous narratives, adding an interactive skills taxonomy, and tightening GitHub/documentation funneling. Keep the current vanilla stack, preserve mobile behavior, and implement a terminal-inspired visual system with accessible dark/light auto support.

**Steps**
1. Phase 1 - Baseline and Information Architecture
1. Audit and normalize top-level page metadata and navigation labels in `index.html` to align with quant branding (hero-first narrative, case-study orientation, explicit academic progression from 3rd-year Physics/Maths to Master’s pathway).
2. Define the featured-project information architecture as `6-10 featured + expandable full repository list` using existing repo inventory (42 public repos) so recruiters see depth without overwhelming first scroll.
3. Keep existing static hosting compatibility (no build pipeline) and confirm no framework migration.

2. Phase 2 - Hero Rebuild (depends on Phase 1)
1. Replace current hero headline/subheadline/description with quant-first pitch and authoritative technical voice.
2. Make `Download CV` the primary CTA and ensure it points to `Armaan Sachdeva-CV.pdf` (verified present in workspace).
3. Add a secondary CTA that routes directly to technical case studies (`#projects` anchor or renamed section label).
4. Shift visual language to terminal-inspired design while retaining readability and academic tone (code-lab accents, restrained motion, high contrast typography).
5. Keep existing mobile nav + smooth scrolling behavior from `script.js` and only adjust text/selector hooks as needed.

3. Phase 3 - PAR Case Study Conversion (depends on Phase 1)
1. Rewrite Black-Hole Ray Tracer, N-body Problem, and PINNs cards into explicit Problem-Action-Result blocks with scannable sub-sections.
2. Integrate MathJax CDN in `index.html` and embed inline/block equations for each PAR case (geodesics, numerical integration schemes, PDE residual terms).
3. Include intuitive analogies directly within Action sections to improve comprehension without diluting rigor.
4. Add media placeholders (GIF/video blocks) with consistent aspect ratio and fallback labels (`Simulation Preview` / `Rendering Pipeline`).
5. Enforce claim integrity: use only verified quantitative results or conservative non-numeric wording where benchmarks are unverified.

4. Phase 4 - Interactive Skills Grid (parallel with Phase 3 after nav/content hooks are set)
1. Replace static 3-card skills list with interactive categorization UI (tabbed filter + responsive card grid).
2. Implement required categories and subskills:
- Physics & Maths: Numerical Integration, PDEs, Lattice Boltzmann
- Machine Learning/AI: PINNs, Convolutional Autoencoders, Optuna
- Software Engineering: C++17 (STL/CMake), React/Next.js, Docker, PostgreSQL
- Scientific Libraries: Uproot, ROOT, LALSuite
3. Add keyboard- and touch-friendly tab interactions in `script.js` (active states, ARIA roles, focus styles).
4. Ensure filtered content remains animation-safe with existing IntersectionObserver logic.

5. Phase 5 - GitHub Synergy and Academic Validation (depends on Phase 3)
1. Update each featured project `View Code` action to deep-link to high-signal files (README, solver entrypoint, experiment script) instead of generic repo roots when possible.
2. Add an expandable `All Repositories` section populated from the discovered repo list; keep concise cards with repo name + topic + link.
3. Embed academic citations in project narratives (e.g., Raissi et al. for PINNs) using external links and compact citation styling.
4. Add a `Research Foundations` micro-section or inline citation callouts to strengthen theoretical credibility.

6. Phase 6 - Theme System, Responsiveness, and Polish (depends on Phases 2-5)
1. Refactor `styles.css` into tokenized variables for terminal-inspired palette plus `prefers-color-scheme` light/dark adaptation.
2. Preserve current breakpoints (`768px`, `480px`) while removing brittle `!important` patterns in touched areas.
3. Tune interaction polish: intentional load-in reveals, minimal but meaningful transitions, no heavy animation clutter.
4. Improve accessibility: visible focus states, semantic heading order, descriptive link labels, reduced-motion fallback.

7. Phase 7 - Verification and QA (depends on all prior phases)
1. Functional checks: mobile menu toggle, anchor smooth scrolling, typing text behavior (if retained), skills filtering interactions, CTA link targets, and contact flow.
2. Math checks: MathJax renders all inline/block equations correctly on desktop/mobile and degrades gracefully if CDN fails.
3. Content checks: all PAR blocks include Problem/Action/Result, all quantitative claims are verified or softened.
4. Link checks: every featured project code button resolves to intended GitHub file path; citation links resolve and open safely.
5. Responsive checks: manual pass at desktop, tablet (`<=768px`), and mobile (`<=480px`) for layout integrity and tap targets.

**Relevant files**
- `c:\Users\singh\OneDrive\Documents\My portfolio website\index.html` — rewrite hero copy/structure, PAR project blocks, MathJax include, citation links, repository expansion section, CTA targets.
- `c:\Users\singh\OneDrive\Documents\My portfolio website\styles.css` — terminal-inspired visual system, theme variables, light/dark adaptation via `prefers-color-scheme`, responsive refinements for new components.
- `c:\Users\singh\OneDrive\Documents\My portfolio website\script.js` — skills tab/filter interactions, any hero text logic updates, compatibility with existing observers and nav behavior.

**Verification**
1. Open the page and verify hero CTA hierarchy and copy readability on first viewport.
2. Confirm `Download CV` opens `Armaan Sachdeva-CV.pdf`.
3. Validate the 3 PAR case studies have visible Problem/Action/Result and rendered equations.
4. Toggle each skills category and verify keyboard + touch behavior.
5. Expand and collapse full repository list and verify selected featured repo deep-links.
6. Verify citation links (including PINNs source) open correctly and match referenced text.
7. Perform mobile checks at `<=768px` and `<=480px` for nav, project cards, and tap spacing.

**Decisions**
- Repository display scope: Hybrid (`6-10 featured + expandable full list`).
- CV target: Existing local file (`Armaan Sachdeva-CV.pdf`).
- Math rendering: MathJax via CDN.
- Aesthetic direction: Terminal-inspired.
- Performance claims policy: No placeholder hard numbers; only verified metrics or conservative phrasing.
- Included scope: hero overhaul, PAR conversion for 3 flagship projects, interactive skills taxonomy, GitHub/citation integration, responsive/theme polish.
- Excluded scope: backend/CMS migration, framework rewrite, automated GitHub API ingestion requiring auth, and unrelated legacy project rewrites outside featured set.

**Further Considerations**
1. Featured set composition recommendation: prioritize Black-Hole Ray Tracer, N-body, PINNs, plus 3-5 supporting projects that demonstrate software engineering depth (e.g., web systems + infra).
2. If a verified benchmark dataset is unavailable, prefer reproducible method statements (`validated against reference trajectories`) over speculative speed/error claims.
3. Consider a later phase to move project data into a JS data object for maintainability after content stabilizes.
