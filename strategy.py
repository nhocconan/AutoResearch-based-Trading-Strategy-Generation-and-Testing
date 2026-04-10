#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 1d volume confirmation and ATR filter
# - Primary signal: Price breaks above/below recent Williams Fractal levels on 4h
# - Volume filter: 1d volume > 1.5x 20-period average volume (ensures institutional participation)
# - ATR filter: 4h ATR(14) < 0.025 * price (low volatility for cleaner breakouts)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 4h
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Williams Fractals require 2-bar confirmation delay (align_htf_to_ltf with additional_delay_bars=2)

name = "4h_1d_williams_fractal_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 4h ATR(14) for volatility filter
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_filter = (atr_14 / close_4h) < 0.025  # ATR < 2.5% of price
    
    # Pre-compute 4h Williams Fractals (requires 2-bar confirmation)
    # Williams Fractal: bearish = high[n-2] < high[n] and high[n-1] < high[n] and high[n+1] < high[n] and high[n+2] < high[n]
    #                bullish = low[n-2] > low[n] and low[n-1] > low[n] and low[n+1] > low[n] and low[n+2] > low[n]
    high = prices['high'].values
    low = prices['low'].values
    
    # Initialize fractal arrays
    bearish_fractal = np.full(n, np.nan)
    bullish_fractal = np.full(n, np.nan)
    
    # Calculate Williams Fractals (need 2 bars on each side)
    for i in range(2, n-2):
        # Bearish fractal: highest high in middle with lower highs on both sides
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish_fractal[i] = high[i]
        
        # Bullish fractal: lowest low in middle with higher lows on both sides
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish_fractal[i] = low[i]
    
    # Align fractals with 2-bar confirmation delay (needed for Williams Fractals)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(atr_filter[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below bullish fractal OR stoploss hit
            if close_4h[i] < bullish_fractal_aligned[i] or close_4h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above bearish fractal OR stoploss hit
            if close_4h[i] > bearish_fractal_aligned[i] or close_4h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams Fractal breakouts with volume and volatility filters
            if vol_spike_aligned[i] and atr_filter[i]:
                # Long: price breaks above bearish fractal (resistance)
                if close_4h[i] > bearish_fractal_aligned[i]:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: price breaks below bullish fractal (support)
                elif close_4h[i] < bullish_fractal_aligned[i]:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals