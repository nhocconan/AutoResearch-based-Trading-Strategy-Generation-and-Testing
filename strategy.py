#!/usr/bin/env python3
name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend filter (primary HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA50 trend
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    trend_up = close > ema_50_4h_aligned
    trend_down = close < ema_50_4h_aligned
    
    # Camarilla levels from previous 4h
    close_prev_4h = df_4h['close'].values
    high_prev_4h = df_4h['high'].values
    low_prev_4h = df_4h['low'].values
    range_prev_4h = high_prev_4h - low_prev_4h
    # R3 and S3 levels (stronger levels)
    r3 = close_prev_4h + range_prev_4h * 1.1 / 4
    s3 = close_prev_4h - range_prev_4h * 1.1 / 4
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # 1d volume filter: current volume > 2.0x 20-period average (on 1h data)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_surge = volume > (2.0 * vol_ma_20)
    
    # Session filter: 08-20 UTC (already datetime64[ms] in index)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # ~6 hours (6*1h) to reduce trade frequency
    
    start_idx = max(20, 1)  # Ensure enough data for volume and Camarilla
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
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
        
        if position == 0 and bars_since_last_trade >= cooldown_bars and in_session[i]:
            # Long: price breaks above R3 with volume surge in 4h uptrend
            if (close[i] > r3_aligned[i] and 
                trending_up and 
                vol_surge[i]):
                signals[i] = 0.20
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below S3 with volume surge in 4h downtrend
            elif (close[i] < s3_aligned[i] and 
                  trending_down and 
                  vol_surge[i]):
                signals[i] = -0.20
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price breaks below S3 or 4h trend changes to down
            if close[i] < s3_aligned[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price breaks above R3 or 4h trend changes to up
            if close[i] > r3_aligned[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: Breakout at 4h Camarilla R3/S3 with volume surge and 4h trend filter works in both bull and bear markets.
# In bull markets: 4h trend up, breakouts above R3 capture continuation.
# In bear markets: 4h trend down, breakdowns below S3 capture continuation.
# Volume surge confirms institutional participation. Using R3/S3 (stronger than R1/S1) increases signal quality.
# 1h timeframe with session filter (08-20 UTC) and cooldown of 6 bars targets 15-37 trades per year.
# Position size 0.20 balances risk and return. Focus on BTC/ETH as primary assets.