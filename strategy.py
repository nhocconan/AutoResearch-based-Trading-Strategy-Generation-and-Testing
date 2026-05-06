#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels with 1d trend filter and volume confirmation
# - Uses weekly Camarilla pivot levels (R3/S3, R4/S4) for institutional support/resistance
# - Uses 1d EMA50 for trend direction filter (long above EMA50, short below)
# - Uses 6h volume spike for entry confirmation
# - Enters long when price breaks above weekly R3 with 1d uptrend and volume
# - Enters short when price breaks below weekly S3 with 1d downtrend and volume
# - Exits when price returns to weekly pivot point (PP) or opposite S3/R3 level
# - Designed to capture institutional level breaks with trend alignment
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_WeeklyCamarilla_R3S3_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Weekly pivot point (PP)
    PP = (high_w + low_w + close_w) / 3
    R1 = PP + (high_w - low_w) * 1.1 / 12
    S1 = PP - (high_w - low_w) * 1.1 / 12
    R2 = PP + (high_w - low_w) * 1.1 / 6
    S2 = PP - (high_w - low_w) * 1.1 / 6
    R3 = PP + (high_w - low_w) * 1.1 / 4
    S3 = PP - (high_w - low_w) * 1.1 / 4
    R4 = PP + (high_w - low_w) * 1.1 / 2
    S4 = PP - (high_w - low_w) * 1.1 / 2
    
    # Align weekly Camarilla levels to 6h timeframe
    PP_6h = align_htf_to_ltf(prices, df_1w, PP)
    R3_6h = align_htf_to_ltf(prices, df_1w, R3)
    S3_6h = align_htf_to_ltf(prices, df_1w, S3)
    R4_6h = align_htf_to_ltf(prices, df_1w, R4)
    S4_6h = align_htf_to_ltf(prices, df_1w, S4)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(PP_6h[i]) or np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above weekly R3 with 1d uptrend and volume
            if close[i] > R3_6h[i] and close[i] > ema_50_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S3 with 1d downtrend and volume
            elif close[i] < S3_6h[i] and close[i] < ema_50_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly PP or breaks below S3
            if close[i] < PP_6h[i] or close[i] < S3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly PP or breaks above R3
            if close[i] > PP_6h[i] or close[i] > R3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals