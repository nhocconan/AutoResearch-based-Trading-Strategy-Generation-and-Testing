#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation
# Uses Camarilla pivot levels for precise intraday breakout entries, aligned with 4h trend via EMA34.
# Volume confirmation filters false breakouts. Session filter (08-20 UTC) reduces noise.
# Designed for 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
# Works in bull markets via upward R3 breakouts and bear markets via downward S3 breakdowns.

name = "1h_Camarilla_R3S3_4hEMA34_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla pivot levels using previous day's OHLC
    # Need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    # Based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    R2 = prev_close + (prev_high - prev_low) * 1.1 / 6
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    PP = (prev_high + prev_low + prev_close) / 3
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    S2 = prev_close - (prev_high - prev_low) * 1.1 / 6
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align daily Camarilla levels to 1h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: 24-period EMA on 1h
    vol_ema_24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous day's pivot
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(vol_ema_24[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_24[i])
        
        if position == 0:
            # Long: price breaks above R3 in uptrend alignment with volume spike
            if close[i] > R3_aligned[i] and ema_34_4h_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 in downtrend alignment with volume spike
            elif close[i] < S3_aligned[i] and ema_34_4h_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below R1 or loses uptrend alignment
            if close[i] < R1_aligned[i] or ema_34_4h_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above S1 or loses downtrend alignment
            if close[i] > S1_aligned[i] or ema_34_4h_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals