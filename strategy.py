#!/usr/bin/env python3
name = "6h_Aroon_Trend_With_D1_Pullback_Entry"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1. Get daily data for trend filter and pullback levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # 14-day Aroon to detect strong trends
    high_14 = df_1d['high'].rolling(window=14, min_periods=14).max().values
    low_14 = df_1d['low'].rolling(window=14, min_periods=14).min().values
    aroon_up = ((14 - (df_1d['high'].rolling(window=14, min_periods=14).apply(lambda x: np.argmax(x)))) / 14 * 100).values
    aroon_down = ((14 - (df_1d['low'].rolling(window=14, min_periods=14).apply(lambda x: np.argmin(x)))) / 14 * 100).values
    
    # Aroon crossover signals: strong uptrend when Aroon-Up > 70 and Aroon-Down < 30
    strong_uptrend = (aroon_up > 70) & (aroon_down < 30)
    strong_downtrend = (aroon_down > 70) & (aroon_up < 30)
    
    # 20-period EMA for pullback entry
    ema_20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align Aroon and EMA to 6h timeframe
    strong_uptrend_aligned = align_htf_to_ltf(prices, df_1d, strong_uptrend)
    strong_downtrend_aligned = align_htf_to_ltf(prices, df_1d, strong_downtrend)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume filter: current volume > 1.5x 20-period average (5 days for 6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day for 6h to reduce trades
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(strong_uptrend_aligned[i]) or 
            np.isnan(strong_downtrend_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or 
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
        is_uptrend = strong_uptrend_aligned[i]
        is_downtrend = strong_downtrend_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Pullback to EMA20 in strong uptrend with volume
            if (close[i] <= ema_20_1d_aligned[i] * 1.005 and  # Within 0.5% above EMA
                close[i] >= ema_20_1d_aligned[i] * 0.995 and  # Within 0.5% below EMA
                is_uptrend and
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Pullback to EMA20 in strong downtrend with volume
            elif (close[i] <= ema_20_1d_aligned[i] * 1.005 and
                  close[i] >= ema_20_1d_aligned[i] * 0.995 and
                  is_downtrend and
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Trend ends or price moves 2% away from EMA
            if not is_uptrend or close[i] > ema_20_1d_aligned[i] * 1.02:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Trend ends or price moves 2% away from EMA
            if not is_downtrend or close[i] < ema_20_1d_aligned[i] * 0.98:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Aroon identifies strong daily trends, then we enter on 6h pullbacks to the 20-day EMA.
# Works in bull markets (buy pullbacks in uptrends) and bear markets (sell rallies in downtrends).
# The Aroon indicator (period 14) identifies when a strong trend is present (>70 up, <30 down).
# Entry occurs when price pulls back to the 20-day EMA within a tight band (±0.5%).
# Volume confirmation ensures institutional participation.
# Exit when the trend weakens or price moves 2% away from EMA.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.