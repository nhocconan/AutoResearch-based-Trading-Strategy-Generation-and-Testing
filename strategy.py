#!/usr/bin/env python3
name = "1d_Keltner_Channel_Breakout_1wTrend"
timeframe = "1d"
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
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    trend_up = close > ema_50_1w_aligned
    trend_down = close < ema_50_1w_aligned
    
    # Daily Keltner Channel (20, 2.0)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.absolute(high - np.concatenate([[np.nan], close[:-1]])), np.absolute(low - np.concatenate([[np.nan], close[:-1]])))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_20 + 2.0 * atr
    kc_lower = ema_20 - 2.0 * atr
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_surge = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~3 days to reduce trade frequency
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
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
        
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price breaks above Keltner upper with volume surge in weekly uptrend
            if (close[i] > kc_upper[i] and 
                trending_up and 
                vol_surge[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below Keltner lower with volume surge in weekly downtrend
            elif (close[i] < kc_lower[i] and 
                  trending_down and 
                  vol_surge[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price breaks below Keltner lower or weekly trend changes to down
            if close[i] < kc_lower[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Keltner upper or weekly trend changes to up
            if close[i] > kc_upper[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Keltner Channel breakout with weekly trend filter captures trending moves while avoiding whipsaws.
# Long when price breaks above daily Keltner upper band with volume surge and weekly uptrend.
# Short when price breaks below daily Keltner lower band with volume surge and weekly downtrend.
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
# Volume surge confirms institutional participation in the breakout.
# Keltner Channel (ATR-based) adapts to volatility better than fixed percentage bands.
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.