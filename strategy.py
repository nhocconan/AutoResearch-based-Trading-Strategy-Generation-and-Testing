# 6h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla R3/S3 breakouts on 6h with weekly trend filter and volume confirmation
# - Camarilla R3/S3 represent key support/resistance levels from previous day
# - Breakout above R3 in weekly uptrend (price > EMA50 weekly) signals bullish continuation
# - Breakdown below S3 in weekly downtrend (price < EMA50 weekly) signals bearish continuation
# - Volume confirmation (2x average) reduces false breakouts
# - Exit when price returns to weekly EMA50 or opposite Camarilla level
# - Weekly trend filter avoids counter-trend trades in choppy markets
# - Target: 15-30 trades/year to stay within optimal range for 6h timeframe
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)

#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    weekly_close = df_1w['close'].values
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Camarilla levels (R3/S3 from previous day)
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    pivot = (d_high + d_low + d_close) / 3
    range_val = d_high - d_low
    r3 = pivot + (range_val * 1.1 / 4)  # R3 level
    s3 = pivot - (range_val * 1.1 / 4)  # S3 level
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike detection (2x 24-period average - 4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_24[i] * 2.0
        
        if position == 0:
            # Long: break above R3 in weekly uptrend with volume
            if close[i] > r3_6h[i] and close[i] > ema_50_6h[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 in weekly downtrend with volume
            elif close[i] < s3_6h[i] and close[i] < ema_50_6h[i] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to weekly EMA50 or breaks below S3
            if close[i] < ema_50_6h[i] or close[i] < s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to weekly EMA50 or breaks above R3
            if close[i] > ema_50_6h[i] or close[i] > r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals