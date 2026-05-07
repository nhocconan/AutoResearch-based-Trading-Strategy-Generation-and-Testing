#!/usr/bin/env python3
name = "6h_Keltner_Channel_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    trend_up = close > ema_200_1w_aligned
    trend_down = close < ema_200_1w_aligned
    
    # Keltner Channel (20, 2.0) on 6h
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i] if not np.isnan(atr[i-1]) else tr[i]
    atr_ma = pd.Series(atr).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_20 + 2.0 * atr_ma
    kc_lower = ema_20 - 2.0 * atr_ma
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day (4*6h) to prevent overtrading
    
    start_idx = max(20, 20)  # Keltner and volume MA need 20 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or 
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
            # Long: Price breaks above Keltner upper with volume in 1w uptrend
            if (close[i] > kc_upper[i] and 
                trending_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Keltner lower with volume in 1w downtrend
            elif (close[i] < kc_lower[i] and 
                  trending_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below Keltner lower or 1w trend changes to down
            if close[i] < kc_lower[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above Keltner upper or 1w trend changes to up
            if close[i] > kc_upper[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 6h timeframe, price breaking above/below Keltner Channel (20,2.0) with volume confirmation and 1-week EMA200 trend filter captures institutional breakout momentum. Keltner channels adapt to volatility, providing dynamic support/resistance. Works in bull markets (breakouts above upper in 1w uptrend) and bear markets (breakdowns below lower in 1w downtrend). Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing significant moves. 1w trend filter ensures alignment with higher timeframe momentum. Volume filter ensures breakouts have institutional participation.