#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_trend_v3
# Hypothesis: 4-hour Camarilla breakout with volume confirmation and trend filter (1d EMA)
# Works in bull/bear by using volatility-adjusted breakouts, volume confirmation, and trend alignment to avoid false signals.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drift.

name = "4h_1d_camarilla_breakout_volume_trend_v3"
timeframe = "4h"
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
    
    # Get daily data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Camarilla levels (based on previous day)
    range_ = prev_high - prev_low
    r4 = prev_close + range_ * 1.1
    s4 = prev_close - range_ * 1.1
    r3 = prev_close + range_ * 1.1 / 2
    s3 = prev_close - range_ * 1.1 / 2
    
    # Daily EMA for trend filter (34-period)
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: close breaks above R4 with volume and above daily EMA
        if (close[i] > r4_aligned[i] and vol_confirm[i] and 
            close[i] > ema_34_aligned[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below S4 with volume and below daily EMA
        elif (close[i] < s4_aligned[i] and vol_confirm[i] and 
              close[i] < ema_34_aligned[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals