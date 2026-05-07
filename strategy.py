#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA34 trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    trend_up = close > ema_34_1d_aligned
    trend_down = close < ema_34_1d_aligned
    
    # Previous 12h close for Camarilla calculation
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels based on previous period
    R4 = prev_close + 1.5 * (high - low)
    R3 = prev_close + 1.0 * (high - low)
    R2 = prev_close + 0.5 * (high - low)
    S2 = prev_close - 0.5 * (high - low)
    S3 = prev_close - 1.0 * (high - low)
    S4 = prev_close - 1.5 * (high - low)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~2 days (4*12h) to reduce trade frequency
    
    start_idx = max(20, 1)  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(R3[i]) or 
            np.isnan(S3[i])):
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
            # Long: price breaks above R3 with volume spike in 1d uptrend
            if (close[i] > R3[i] and 
                trending_up and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below S3 with volume spike in 1d downtrend
            elif (close[i] < S3[i] and 
                  trending_down and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price breaks below S3 or 1d trend changes to down
            if close[i] < S3[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R3 or 1d trend changes to up
            if close[i] > R3[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout captures institutional breakout moves in both bull and bear markets.
# Long when price breaks above Camarilla R3 with volume spike and 1d uptrend.
# Short when price breaks below Camarilla S3 with volume spike and 1d downtrend.
# Works in bull markets (sustained uptrend with breakouts) and bear markets (sustained downtrend with breakdowns).
# Volume spike confirms institutional participation. 12h timeframe reduces noise vs lower TFs.
# Discrete position sizing (0.25) balances risk and minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.