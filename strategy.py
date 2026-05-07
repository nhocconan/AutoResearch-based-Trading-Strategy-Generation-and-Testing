#!/usr/bin/env python3
# Hypothesis: 4h Camarilla pivot reversal with 1d trend filter and volume spike.
# In both bull and bear markets, price tends to reverse from key intraday support/resistance levels
# (Camarilla) when aligned with higher timeframe trend. Volume spike confirms institutional interest.
# Uses discrete position sizing (0.25) to limit turnover. Target: 20-50 trades/year.

name = "4h_Camarilla_R3S3_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Trend Filter (EMA 34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h Camarilla Pivot Levels (using prior 1d OHLC) ===
    df_1d_full = get_htf_data(prices, '1d')
    if len(df_1d_full) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's range
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # We use prior day's data to avoid look-ahead
    prior_close = df_1d_full['close'].shift(1).values
    prior_high = df_1d_full['high'].shift(1).values
    prior_low = df_1d_full['low'].shift(1).values
    
    camarilla_width = (prior_high - prior_low) * 1.1 / 2
    r3 = prior_close + camarilla_width
    s3 = prior_close - camarilla_width
    
    # Align to 4h timeframe (same value throughout the 4h day)
    r3_aligned = align_htf_to_ltf(prices, df_1d_full, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d_full, s3)
    
    # === Volume Spike Detector ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Session Filter: 08:00-20:00 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(close[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period average (strict filter)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price rejects S3 support in 1d uptrend with volume spike
            if (low[i] <= s3_aligned[i] and  # touched or pierced S3
                close[i] > s3_aligned[i] and   # closed back above S3 (rejection)
                close[i] > ema_34_1d_aligned[i] and  # 1d uptrend
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price rejects R3 resistance in 1d downtrend with volume spike
            elif (high[i] >= r3_aligned[i] and  # touched or pierced R3
                  close[i] < r3_aligned[i] and   # closed back below R3 (rejection)
                  close[i] < ema_34_1d_aligned[i] and  # 1d downtrend
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to opposite Camarilla level or trend fails
            if position == 1:
                # Exit long if price reaches R3 or trend turns down
                if (high[i] >= r3_aligned[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if price reaches S3 or trend turns up
                if (low[i] <= s3_aligned[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals