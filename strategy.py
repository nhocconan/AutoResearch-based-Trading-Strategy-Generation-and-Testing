#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1-day volume confirmation and 1-week trend filter
# Long when price breaks above R3 AND daily volume > 1.5x 20-day average AND weekly close > weekly open
# Short when price breaks below S3 AND daily volume > 1.5x 20-day average AND weekly close < weekly open
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 6h timeframe
# Camarilla levels from daily data provide institutional support/resistance; volume confirms breakout strength;
# weekly trend filter ensures alignment with higher-timeframe momentum, reducing false breakouts in ranging markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    r4 = pivot + (range_1d * 1.1)
    s4 = pivot - (range_1d * 1.1)
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === Daily volume confirmation (1.5x 20-day average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # === Weekly trend filter (close > open for uptrend, close < open for downtrend) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    weekly_uptrend = close_1w > open_1w  # True for bullish weekly candle
    weekly_downtrend = close_1w < open_1w  # True for bearish weekly candle
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        # Exit conditions: reverse signal or volatility exhaustion
        if position == 1 and (price < s3_aligned[i] or not vol_confirm):
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and (price > r3_aligned[i] or not vol_confirm):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic (only when flat)
        if position == 0:
            # Long when: price breaks above R3 AND volume confirmation AND weekly uptrend
            if price > r3_aligned[i] and vol_confirm and weekly_up:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: price breaks below S3 AND volume confirmation AND weekly downtrend
            elif price < s3_aligned[i] and vol_confirm and weekly_down:
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

name = "6h_CamarillaR3S3_Volume1.5x_WeeklyTrend"
timeframe = "6h"
leverage = 1.0