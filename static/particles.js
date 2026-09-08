// Particle and 3D Halo Crown Animation System (Direct Canvas Rendering)

export class Particle {
    constructor(x, y, emojis = ['💖', '❤️', '💕', '💗', '💓', '💝']) {
        this.x = x;
        this.y = y;
        this.emoji = emojis[Math.floor(Math.random() * emojis.length)];
        this.size = Math.round(Math.random() * 8 + 26);
        this.opacity = 1.0;
        this.fadeRate = Math.random() * 0.024 + 0.022; // ~40 frames life
        this.vx = (Math.random() - 0.5) * 3.6;         // Dispersal velocity
        this.vy = -(Math.random() * 3.4 + 2.4);        // Upward buoyancy
        this.rotation = (Math.random() - 0.5) * 0.4;
        this.angularVel = (Math.random() - 0.5) * 0.05;
        this.scale = 0.5;
        this.targetScale = 1.0;
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.vy += 0.035; // Gentle upward deceleration
        this.rotation += this.angularVel;
        if (this.scale < this.targetScale) {
            this.scale += 0.10;
        }
        this.opacity -= this.fadeRate;
    }

    draw(ctx) {
        if (this.opacity <= 0) return;
        ctx.save();
        ctx.globalAlpha = Math.max(0, this.opacity);
        ctx.translate(this.x, this.y);
        ctx.rotate(this.rotation);
        ctx.scale(this.scale, this.scale);

        ctx.font = `${this.size}px "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(this.emoji, 0, 0);
        ctx.restore();
    }
}

/**
 * Draws a perspective 3D rotating halo crown hovering above the head.
 * Front emojis are scaled up and drawn last; back emojis are scaled down and drawn first.
 */
export function draw3DCrown(ctx, faceCenterX, faceCenterY, faceWidth, faceHeight, crownAngle, emojis) {
    if (!emojis || emojis.length === 0) return;

    const numItems = 6;
    const rx = faceWidth * 0.62;        // Horizontal orbital radius
    const ry = faceHeight * 0.15;       // Vertical orbital radius (perspective tilt)
    const cy = faceCenterY - faceHeight * 0.88; // Floating comfortably above top of head/hair

    // Compute items and sort by depth (Z-order)
    const items = [];
    for (let i = 0; i < numItems; i++) {
        const angle = crownAngle + (i * 2 * Math.PI / numItems);
        const sinA = Math.sin(angle);
        const cosA = Math.cos(angle);
        const hx = faceCenterX + cosA * rx;
        const hy = cy + sinA * ry;
        const depthScale = 0.80 + sinA * 0.28; // Depth perspective: front items (sin > 0) are larger
        const size = Math.round((faceWidth * 0.18) * depthScale);
        const opacity = 0.70 + sinA * 0.30;    // Front items are crisper

        items.push({
            emoji: emojis[i % emojis.length],
            x: hx,
            y: hy,
            size,
            opacity,
            zOrder: sinA
        });
    }

    // Sort: back items first, front items last
    items.sort((a, b) => a.zOrder - b.zOrder);

    for (const item of items) {
        ctx.save();
        ctx.globalAlpha = Math.max(0.1, Math.min(1.0, item.opacity));
        ctx.font = `${item.size}px "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(item.emoji, item.x, item.y);
        ctx.restore();
    }
}
