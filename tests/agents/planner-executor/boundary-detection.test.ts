/**
 * Tests for boundary detection (auth pages, checkout pages).
 */

import {
  detectAuthBoundary,
  detectCheckoutPage,
  isCheckoutElement,
  DEFAULT_AUTH_BOUNDARY_CONFIG,
  DEFAULT_CHECKOUT_CONFIG,
  type AuthBoundaryConfig,
  type CheckoutDetectionConfig,
} from '../../../src/agents/planner-executor/boundary-detection';

describe('boundary-detection', () => {
  describe('DEFAULT_AUTH_BOUNDARY_CONFIG', () => {
    it('should have expected default values', () => {
      expect(DEFAULT_AUTH_BOUNDARY_CONFIG.enabled).toBe(true);
      expect(DEFAULT_AUTH_BOUNDARY_CONFIG.stopOnAuth).toBe(true);
      expect(DEFAULT_AUTH_BOUNDARY_CONFIG.urlPatterns).toContain('/signin');
      expect(DEFAULT_AUTH_BOUNDARY_CONFIG.urlPatterns).toContain('/login');
      expect(DEFAULT_AUTH_BOUNDARY_CONFIG.authSuccessMessage).toBe(
        'Reached authentication boundary (login required)'
      );
    });
  });

  describe('DEFAULT_CHECKOUT_CONFIG', () => {
    it('should have expected default values', () => {
      expect(DEFAULT_CHECKOUT_CONFIG.enabled).toBe(true);
      expect(DEFAULT_CHECKOUT_CONFIG.cartUrlPatterns).toContain('/cart');
      expect(DEFAULT_CHECKOUT_CONFIG.checkoutUrlPatterns).toContain('/checkout');
      expect(DEFAULT_CHECKOUT_CONFIG.checkoutElementPatterns).toContain('proceed to checkout');
    });
  });

  describe('detectAuthBoundary', () => {
    it('should detect signin URL', () => {
      const result = detectAuthBoundary('https://example.com/signin');
      expect(result.isAuthBoundary).toBe(true);
      expect(result.matchedPattern).toBe('/signin');
    });

    it('should detect login URL', () => {
      const result = detectAuthBoundary('https://example.com/login');
      expect(result.isAuthBoundary).toBe(true);
      expect(result.matchedPattern).toBe('/login');
    });

    it('should detect sign-in URL with hyphen', () => {
      const result = detectAuthBoundary('https://example.com/sign-in');
      expect(result.isAuthBoundary).toBe(true);
      expect(result.matchedPattern).toBe('/sign-in');
    });

    it('should detect Amazon signin URL', () => {
      const result = detectAuthBoundary('https://amazon.com/ap/signin');
      expect(result.isAuthBoundary).toBe(true);
      // Note: /signin matches before /ap/signin, which is fine - both are auth pages
      expect(result.matchedPattern).toBeTruthy();
    });

    it('should detect Amazon register URL', () => {
      const result = detectAuthBoundary('https://amazon.com/ap/register');
      expect(result.isAuthBoundary).toBe(true);
      expect(result.matchedPattern).toBe('/ap/register');
    });

    it('should detect Amazon CAPTCHA/claim URL', () => {
      const result = detectAuthBoundary('https://amazon.com/ax/claim');
      expect(result.isAuthBoundary).toBe(true);
      expect(result.matchedPattern).toBe('/ax/claim');
    });

    it('should detect auth URL', () => {
      const result = detectAuthBoundary('https://example.com/auth/callback');
      expect(result.isAuthBoundary).toBe(true);
      expect(result.matchedPattern).toBe('/auth');
    });

    it('should not detect non-auth URL', () => {
      const result = detectAuthBoundary('https://example.com/products');
      expect(result.isAuthBoundary).toBe(false);
      expect(result.matchedPattern).toBeNull();
    });

    it('should be case insensitive', () => {
      const result = detectAuthBoundary('https://example.com/LOGIN');
      expect(result.isAuthBoundary).toBe(true);
    });

    it('should handle empty URL', () => {
      const result = detectAuthBoundary('');
      expect(result.isAuthBoundary).toBe(false);
      expect(result.matchedPattern).toBeNull();
    });

    it('should respect disabled config', () => {
      const config: AuthBoundaryConfig = {
        ...DEFAULT_AUTH_BOUNDARY_CONFIG,
        enabled: false,
      };
      const result = detectAuthBoundary('https://example.com/signin', config);
      expect(result.isAuthBoundary).toBe(false);
    });

    it('should use custom URL patterns', () => {
      const config: AuthBoundaryConfig = {
        ...DEFAULT_AUTH_BOUNDARY_CONFIG,
        urlPatterns: ['/custom-auth'],
      };
      const result = detectAuthBoundary('https://example.com/custom-auth', config);
      expect(result.isAuthBoundary).toBe(true);
      expect(result.matchedPattern).toBe('/custom-auth');
    });
  });

  describe('detectCheckoutPage', () => {
    it('should detect cart URL', () => {
      const result = detectCheckoutPage('https://shop.com/cart');
      expect(result.isCheckoutRelated).toBe(true);
      expect(result.isCart).toBe(true);
      expect(result.isCheckout).toBe(false);
      expect(result.matchedPattern).toBe('/cart');
    });

    it('should detect basket URL', () => {
      const result = detectCheckoutPage('https://shop.com/basket');
      expect(result.isCheckoutRelated).toBe(true);
      expect(result.isCart).toBe(true);
      expect(result.isCheckout).toBe(false);
    });

    it('should detect bag URL', () => {
      const result = detectCheckoutPage('https://shop.com/bag');
      expect(result.isCheckoutRelated).toBe(true);
      expect(result.isCart).toBe(true);
    });

    it('should detect checkout URL', () => {
      const result = detectCheckoutPage('https://shop.com/checkout');
      expect(result.isCheckoutRelated).toBe(true);
      expect(result.isCart).toBe(false);
      expect(result.isCheckout).toBe(true);
      expect(result.matchedPattern).toBe('/checkout');
    });

    it('should detect payment URL', () => {
      const result = detectCheckoutPage('https://shop.com/payment');
      expect(result.isCheckoutRelated).toBe(true);
      expect(result.isCheckout).toBe(true);
    });

    it('should detect order URL', () => {
      const result = detectCheckoutPage('https://shop.com/order');
      expect(result.isCheckoutRelated).toBe(true);
      expect(result.isCheckout).toBe(true);
    });

    it('should detect Amazon cart URL', () => {
      const result = detectCheckoutPage('https://amazon.com/gp/cart');
      expect(result.isCheckoutRelated).toBe(true);
      expect(result.isCart).toBe(true);
    });

    it('should detect Amazon checkout URL', () => {
      const result = detectCheckoutPage('https://amazon.com/gp/checkout');
      expect(result.isCheckoutRelated).toBe(true);
      expect(result.isCheckout).toBe(true);
    });

    it('should not detect regular product URL', () => {
      const result = detectCheckoutPage('https://shop.com/products/widget');
      expect(result.isCheckoutRelated).toBe(false);
      expect(result.isCart).toBe(false);
      expect(result.isCheckout).toBe(false);
      expect(result.matchedPattern).toBeNull();
    });

    it('should be case insensitive', () => {
      const result = detectCheckoutPage('https://shop.com/CHECKOUT');
      expect(result.isCheckoutRelated).toBe(true);
      expect(result.isCheckout).toBe(true);
    });

    it('should handle empty URL', () => {
      const result = detectCheckoutPage('');
      expect(result.isCheckoutRelated).toBe(false);
    });

    it('should respect disabled config', () => {
      const config: CheckoutDetectionConfig = {
        ...DEFAULT_CHECKOUT_CONFIG,
        enabled: false,
      };
      const result = detectCheckoutPage('https://shop.com/checkout', config);
      expect(result.isCheckoutRelated).toBe(false);
    });

    it('should prioritize cart over checkout patterns', () => {
      // Cart patterns are checked first
      const config: CheckoutDetectionConfig = {
        ...DEFAULT_CHECKOUT_CONFIG,
        cartUrlPatterns: ['/cart'],
        checkoutUrlPatterns: ['/cart'], // Same pattern
      };
      const result = detectCheckoutPage('https://shop.com/cart', config);
      expect(result.isCart).toBe(true);
      expect(result.isCheckout).toBe(false);
    });
  });

  describe('isCheckoutElement', () => {
    it('should detect "proceed to checkout" text', () => {
      expect(isCheckoutElement('Proceed to Checkout')).toBe(true);
    });

    it('should detect "go to checkout" text', () => {
      expect(isCheckoutElement('Go to Checkout')).toBe(true);
    });

    it('should detect "view cart" text', () => {
      expect(isCheckoutElement('View Cart')).toBe(true);
    });

    it('should detect "shopping cart" text', () => {
      expect(isCheckoutElement('Shopping Cart (3 items)')).toBe(true);
    });

    it('should detect "your cart" text', () => {
      expect(isCheckoutElement('Your Cart')).toBe(true);
    });

    it('should detect "sign in to checkout" text', () => {
      expect(isCheckoutElement('Sign in to checkout')).toBe(true);
    });

    it('should detect "continue to payment" text', () => {
      expect(isCheckoutElement('Continue to Payment')).toBe(true);
    });

    it('should detect "place your order" text', () => {
      expect(isCheckoutElement('Place Your Order')).toBe(true);
    });

    it('should detect "buy now" text', () => {
      expect(isCheckoutElement('Buy Now')).toBe(true);
    });

    it('should not match regular text', () => {
      expect(isCheckoutElement('Add to Wishlist')).toBe(false);
    });

    it('should not match unrelated button text', () => {
      expect(isCheckoutElement('Continue Shopping')).toBe(false);
    });

    it('should be case insensitive', () => {
      expect(isCheckoutElement('PROCEED TO CHECKOUT')).toBe(true);
    });

    it('should handle empty text', () => {
      expect(isCheckoutElement('')).toBe(false);
    });

    it('should handle null-like text', () => {
      expect(isCheckoutElement(null as unknown as string)).toBe(false);
    });

    it('should use custom patterns', () => {
      const config: CheckoutDetectionConfig = {
        ...DEFAULT_CHECKOUT_CONFIG,
        checkoutElementPatterns: ['custom checkout'],
      };
      expect(isCheckoutElement('Custom Checkout', config)).toBe(true);
      expect(isCheckoutElement('Proceed to Checkout', config)).toBe(false);
    });
  });
});
