#!/usr/bin/env python3
name = "12h_Donchian20_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 22:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA34 trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    trend_up = close > ema_34_1w_aligned
    trend_down = close < ema_34_1w_aligned
    
    # Donchian(20) on 12h
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 1  # ~12 hours (1*12h) to reduce trade frequency
    
    start_idx = max(1, 20)  # Ensure enough data for Donchian and volume
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
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
            # Long: breakout above Donchian high with volume spike in 1w uptrend
            if (close[i] > donch_high[i] and 
                trending_up and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: breakdown below Donchian low with volume spike in 1w downtrend
            elif (close[i] < donch_low[i] and 
                  trending_down and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: close below Donchian low or 1w trend changes to down
            if close[i] < donch_low[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close above Donchian high or 1w trend changes to up
            if close[i] > donch_high[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout with volume confirmation and 1w trend filter.
# Long when price breaks above 20-period high with volume spike in 1w uptrend.
# Short when price breaks below 20-period low with volume spike in 1w downtrend.
# Works in bull markets (sustained 1w uptrend) and bear markets (sustained 1w downtrend).
# Volume spike confirms institutional participation. 12h timeframe reduces noise and trade frequency.
# Discrete position sizing (0.25) balances risk and minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.