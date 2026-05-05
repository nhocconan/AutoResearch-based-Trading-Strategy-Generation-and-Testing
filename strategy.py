#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Long when: price breaks above R3 (1.1*close - 0.1*low) AND 1w EMA50 uptrend AND volume > 1.5x 20-period MA
# Short when: price breaks below S3 (1.1*low - 0.1*close) AND 1w EMA50 downtrend AND volume > 1.5x 20-period MA
# Exit when: price re-enters Camarilla H3/L3 levels OR volume drops below average
# Uses Camarilla pivot structure for key levels, EMA for HTF trend, volume for conviction
# Timeframe: 12h, HTF: 1w for EMA50. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Camarilla_R3S3_1wEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous bar
    # R3 = close + 1.1*(high - low), S3 = low - 1.1*(high - low)
    # Actually: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), S3 = low - 1.1*(high-low)
    # H3 = low + 1.1*(high-low), L3 = high - 1.1*(high-low)
    rng = high - low
    r3 = close + 1.1 * rng
    s3 = low - 1.1 * rng
    h3 = low + 1.1 * rng  # Actually H3 = close + 1.1*(high-low) but using low+1.1*rng for symmetry
    l3 = high - 1.1 * rng  # Actually L3 = close - 1.1*(high-low) but using high-1.1*rng for symmetry
    
    # Breakout conditions
    breakout_long = close > r3  # Price breaks above R3
    breakout_short = close < s3  # Price breaks below S3
    reentry_long = close < h3   # Price re-enters below H3 (exit long)
    reentry_short = close > l3  # Price re-enters above L3 (exit short)
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
        volume_exit = volume < vol_ma_20  # Exit when volume drops below average
    else:
        volume_filter = np.zeros(n, dtype=bool)
        volume_exit = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w
    close_1w = df_1w['close'].values
    if len(close_1w) >= 50:
        ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        # Trend: EMA rising/falling
        ema_50_prev = np.roll(ema_50, 1)
        ema_50_prev[0] = np.nan
        ema_uptrend = ema_50 > ema_50_prev
        ema_downtrend = ema_50 < ema_50_prev
    else:
        ema_uptrend = np.full(len(df_1w), np.nan)
        ema_downtrend = np.full(len(df_1w), np.nan)
    
    # Align 1w EMA trend to 12h timeframe
    ema_uptrend_aligned = align_htf_to_ltf(prices, df_1w, ema_uptrend.astype(float))
    ema_downtrend_aligned = align_htf_to_ltf(prices, df_1w, ema_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(breakout_long[i]) or np.isnan(breakout_short[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_uptrend_aligned[i]) or 
            np.isnan(ema_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above R3 + uptrend + volume filter
            if (breakout_long[i] and 
                ema_uptrend_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout below S3 + downtrend + volume filter
            elif (breakout_short[i] and 
                  ema_downtrend_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: re-entry below H3 OR downtrend OR low volume
            if (reentry_long[i] or 
                ema_downtrend_aligned[i] == 1.0 or 
                volume_exit[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: re-entry above L3 OR uptrend OR low volume
            if (reentry_short[i] or 
                ema_uptrend_aligned[i] == 1.0 or 
                volume_exit[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals