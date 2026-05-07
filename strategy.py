#!/usr/bin/env python3
name = "1d_Williams_Fractal_Breakout_1wTrend_VolumeSurge_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter using EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    trend_up = close > ema_34_1w_aligned
    trend_down = close < ema_34_1w_aligned
    
    # Daily Williams Fractals (requires 2-bar confirmation)
    from mtf_data import compute_williams_fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    
    # Volume surge: current volume > 1.5x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_surge = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~3 days to reduce trade frequency
    
    start_idx = max(20, 2)  # Need 20 for volume MA and 2 for fractals
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: bullish fractal break with volume surge in weekly uptrend
            if bullish_fractal_aligned[i] and trending_up and vol_surge[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: bearish fractal break with volume surge in weekly downtrend
            elif bearish_fractal_aligned[i] and trending_down and vol_surge[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: bearish fractal or weekly trend changes to down
            if bearish_fractal_aligned[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish fractal or weekly trend changes to up
            if bullish_fractal_aligned[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams Fractal breakouts with volume surge and weekly trend filter work in both bull and bear markets.
# In bull markets: weekly trend up, bullish fractal breaks capture continuation.
# In bear markets: weekly trend down, bearish fractal breaks capture continuation.
# Volume surge confirms institutional participation. Fractals provide natural support/resistance levels.
# Using 1d timeframe with weekly trend filter targets 30-100 trades over 4 years (7-25/year).
# Cooldown of 3 days and position size 0.25 minimize fee churn while maintaining signal quality.