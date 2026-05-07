#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# 6h_Camarilla_R3_S3_Fade_1dTrend_Volume
# Hypothesis: On 6h chart, fade at Camarilla R3/S3 levels when 1d trend is strong and volume confirms.
# In strong trends (1d EMA50 aligned), price often pulls back to R3/S3 before continuing.
# Fade these levels with limit orders: sell at R3 in uptrend, buy at S3 in downtrend.
# Uses volume spike to confirm institutional interest at these levels.
# Designed for moderate trade frequency (~15-30/year) to work in both bull and bear markets.
# Works in bull: sells at R3 during pullbacks in uptrend.
# Works in bear: buys at S3 during bounces in downtrend.
timeframe = "6h"
name = "6h_Camarilla_R3_S3_Fade_1dTrend_Volume"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous day's range
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Wait for warmup
        # Skip if any critical value is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Fade at R3 in uptrend: sell when price reaches R3 during 1d uptrend
            if (ema_1d_aligned[i] > ema_1d_aligned[i-1] and  # 1d uptrend
                high[i] >= camarilla_r3_aligned[i] and       # price touches or exceeds R3
                volume[i] > 2.0 * vol_ma[i]):                # volume confirmation
                signals[i] = -0.25  # short
                position = -1
            # Fade at S3 in downtrend: buy when price reaches S3 during 1d downtrend
            elif (ema_1d_aligned[i] < ema_1d_aligned[i-1] and  # 1d downtrend
                  low[i] <= camarilla_s3_aligned[i] and        # price touches or goes below S3
                  volume[i] > 2.0 * vol_ma[i]):                # volume confirmation
                signals[i] = 0.25   # long
                position = 1
        elif position == 1:
            # Exit long: price reaches S3 (opposite level) or trend changes
            if (low[i] <= camarilla_s3_aligned[i] or 
                ema_1d_aligned[i] > ema_1d_aligned[i-1]):  # trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches R3 (opposite level) or trend changes
            if (high[i] >= camarilla_r3_aligned[i] or 
                ema_1d_aligned[i] < ema_1d_aligned[i-1]):  # trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals