#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h volume spike and 1d trend filter
# Long when price breaks above R1 AND 4h volume > 2x 20-period volume EMA AND 1d close > 1d EMA50 (uptrend)
# Short when price breaks below S1 AND 4h volume > 2x 20-period volume EMA AND 1d close < 1d EMA50 (downtrend)
# Uses 1h for precise entry timing, 4h for volume confirmation, 1d for trend direction
# Session filter (08-20 UTC) to avoid low-liquidity periods
# Discrete sizing: 0.20 to balance return and fee drag
# Target: 60-150 total trades over 4 years = 15-37/year for 1h

name = "1h_Camarilla_R1S1_4hVolSpike_1dTrend_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1h Camarilla levels from previous day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 4h volume spike filter (volume > 2x 20-period volume EMA)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    vol_4h = df_4h['volume'].values
    vol_ema_20_4h = pd.Series(vol_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_4h = vol_4h > (vol_ema_20_4h * 2.0)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h.astype(float))
    
    # 1d trend filter (EMA50)
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_spike_4h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1 AND 4h volume spike AND 1d uptrend
            if (close[i] > r1_aligned[i] and 
                volume_spike_4h_aligned[i] > 0.5 and 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S1 AND 4h volume spike AND 1d downtrend
            elif (close[i] < s1_aligned[i] and 
                  volume_spike_4h_aligned[i] > 0.5 and 
                  downtrend_1d_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 OR 1d trend changes to downtrend
            if (close[i] < s1_aligned[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R1 OR 1d trend changes to uptrend
            if (close[i] > r1_aligned[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals