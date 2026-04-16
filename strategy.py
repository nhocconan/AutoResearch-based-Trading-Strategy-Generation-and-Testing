#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal with 1d trend filter and volume confirmation
# Williams Fractals identify potential reversal points. In trending markets (1d EMA50),
# we trade breakouts of these fractal levels in the direction of the trend.
# Volume confirms breakout strength. Designed to work in both bull (trend-following)
# and bear (mean-reversion at extremes) markets via trend filter.
# Target: 50-150 total trades over 4 years (~12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (HTF for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Williams Fractals on 6h (requires 5-bar window) ===
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-2] and high[n+1] < high[n]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-2] and low[n+1] > low[n]
    # We'll compute the fractal values and shift by 2 to avoid look-ahead (needs 2 future bars)
    n1 = n - 2
    bearish_fractal = np.full(n, np.nan)
    bullish_fractal = np.full(n, np.nan)
    
    for i in range(2, n-2):
        if (high[i-2] < high[i-1] and 
            high[i] > high[i-1] and 
            high[i-3] < high[i-2] and 
            high[i+1] < high[i]):
            bearish_fractal[i] = high[i]
        if (low[i-2] > low[i-1] and 
            low[i] < low[i-1] and 
            low[i-3] > low[i-2] and 
            low[i+1] > low[i]):
            bullish_fractal[i] = low[i]
    
    # Align fractals with 2-bar delay (Williams needs confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    
    # === Volume confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_50 = ema_50_1d_aligned[i]
        bear_fract = bearish_fractal_aligned[i]
        bull_fract = bullish_fractal_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below bullish fractal (support break)
            if not np.isnan(bull_fract) and price < bull_fract:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above bearish fractal (resistance break)
            if not np.isnan(bear_fract) and price > bear_fract:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade on volume spike (avoid chop)
            if vol_ratio_val > 1.8:
                # UPTREND: price > EMA50 -> look for bullish fractal breakout
                if price > ema_50:
                    # Buy when price breaks above bearish fractal (resistance) with volume
                    if not np.isnan(bear_fract) and price > bear_fract:
                        signals[i] = 0.25
                        position = 1
                        continue
                # DOWNTREND: price < EMA50 -> look for bearish fractal breakdown
                else:
                    # Sell when price breaks below bullish fractal (support) with volume
                    if not np.isnan(bull_fract) and price < bull_fract:
                        signals[i] = -0.25
                        position = -1
                        continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsFractal_EMA50_Volume"
timeframe = "6h"
leverage = 1.0