#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_Momentum
# Hypothesis: 6s donchian breakout (14) filtered by weekly pivot direction and volume surge.
# Long when price breaks above donchian high(14), weekly pivot is bullish (close > pivot), volume > 2x avg.
# Short when price breaks below donchian low(14), weekly pivot is bearish (close < pivot), volume > 2x avg.
# Exit when price returns to donchian midpoint or weekly pivot flips.
# Weekly pivot provides structural bias, donchian captures breakouts, volume confirms strength.
# Works in bull/bear: pivot filters counter-trend, donchian catches momentum, volume avoids chop.

name = "6h_WeeklyPivot_DonchianBreakout_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Donchian(14) channels ---
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    for i in range(14, n):
        donchian_high[i] = np.max(high[i-14:i])
        donchian_low[i] = np.min(low[i-14:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # --- Weekly Pivot (P) and bias ---
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    pivot_w = (high_w + low_w + close_w) / 3
    bullish_w = close_w > pivot_w  # True if weekly close above pivot
    
    # Align weekly pivot bias to 6h
    bullish_w_aligned = align_htf_to_ltf(prices, df_weekly, bullish_w.astype(float))
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Donchian(14), volume MA(20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or
            np.isnan(bullish_w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 2.0  # 100% above average
        
        if position == 0:
            if breakout_up and bullish_w_aligned[i] > 0.5 and vol_spike:
                # Long: upward breakout + bullish weekly pivot + volume spike
                signals[i] = 0.25
                position = 1
            elif breakout_down and bullish_w_aligned[i] < 0.5 and vol_spike:
                # Short: downward breakout + bearish weekly pivot + volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls to midpoint OR weekly pivot turns bearish
                if close[i] < donchian_mid[i] or bullish_w_aligned[i] < 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises to midpoint OR weekly pivot turns bullish
                if close[i] > donchian_mid[i] or bullish_w_aligned[i] > 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals