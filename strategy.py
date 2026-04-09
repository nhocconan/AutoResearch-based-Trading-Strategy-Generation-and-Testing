#!/usr/bin/env python3
# 4h_12h_camarilla_pullback_v1
# Hypothesis: Price pullback to Camarilla levels on 12h chart during 12h uptrend/downtrend.
# Long when price touches S3 level in uptrend, short when price touches R3 level in downtrend.
# Uses 12h EMA(20) for trend filter and volume confirmation.
# Works in bull/bear markets by following 12h trend. Low trade frequency due to specific level touches.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_pullback_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (based on previous day's range)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_r3 = np.full(len(close_12h), np.nan)
    camarilla_s3 = np.full(len(close_12h), np.nan)
    camarilla_r4 = np.full(len(close_12h), np.nan)
    camarilla_s4 = np.full(len(close_12h), np.nan)
    
    for i in range(1, len(close_12h)):
        # Previous period's range
        prev_high = high_12h[i-1]
        prev_low = low_12h[i-1]
        prev_close = close_12h[i-1]
        range_ = prev_high - prev_low
        
        camarilla_r3[i] = prev_close + range_ * 1.1 / 4
        camarilla_s3[i] = prev_close - range_ * 1.1 / 4
        camarilla_r4[i] = prev_close + range_ * 1.1 / 2
        camarilla_s4[i] = prev_close - range_ * 1.1 / 2
    
    # Calculate 12h EMA(20) for trend filter
    ema_12h = np.zeros_like(close_12h, dtype=float)
    ema_12h[0] = close_12h[0]
    alpha = 2.0 / (20 + 1)
    for i in range(1, len(close_12h)):
        ema_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_12h[i-1]
    
    # Align 12h indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.3
        
        # Trend filter: price above/below 12h EMA
        trend_up = close[i] > ema_12h_aligned[i]
        trend_down = close[i] < ema_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 or stops loss
            if close[i] < camarilla_s3_aligned[i] or close[i] < prices['close'].iloc[i-1] - 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 or stops loss
            if close[i] > camarilla_r3_aligned[i] or close[i] > prices['close'].iloc[i-1] + 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches S3 level in uptrend with volume confirmation
            if (abs(close[i] - camarilla_s3_aligned[i]) < 0.001 * close[i] and 
                trend_up and vol_ok):
                position = 1
                signals[i] = 0.25
            # Enter short: price touches R3 level in downtrend with volume confirmation
            elif (abs(close[i] - camarilla_r3_aligned[i]) < 0.001 * close[i] and 
                  trend_down and vol_ok):
                position = -1
                signals[i] = -0.25
    
    return signals