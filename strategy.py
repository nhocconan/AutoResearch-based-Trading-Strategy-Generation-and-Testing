#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA50 trend filter and volume spike.
Long when price breaks above R1 in 4h uptrend with volume > 2x 20-period MA.
Short when price breaks below S1 in 4h downtrend with volume spike.
Camarilla levels provide intraday support/resistance; 4h trend filters false breakouts.
Uses discrete position size 0.20 to minimize fee churn. Targets 15-37 trades/year on 1h.
Session filter (08-20 UTC) to avoid low-volume Asian session noise.
Works in bull/bear by following 4h trend; Camarilla adapts to volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    
    # Get 4h data for trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # need 20 for EMA50
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    uptrend_4h = close_4h > ema_50_4h
    downtrend_4h = close_4h < ema_50_4h
    
    # Align 4h trend to 1h
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h)
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h)
    
    # Calculate Camarilla levels on 4h (based on previous 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_prev = df_4h['close'].shift(1).values  # previous close for Camarilla
    
    # Camarilla R1, S1, R3, S3 (using previous 4h bar)
    range_4h = high_4h - low_4h
    r1 = close_4h_prev + range_4h * 1.0/12
    s1 = close_4h_prev - range_4h * 1.0/12
    r3 = close_4h_prev + range_4h * 3.0/12
    s3 = close_4h_prev - range_4h * 3.0/12
    
    # Align Camarilla levels to 1h (use previous 4h bar's levels)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Volume confirmation: volume > 2.0x 20-period MA on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for EMA, 20 for volume MA, 1 for Camarilla)
    start_idx = max(20, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(uptrend_4h_aligned[i]) or np.isnan(downtrend_4h_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten or hold flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.0  # exit long
                position = 0
            else:
                signals[i] = 0.0  # exit short
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 4h uptrend and volume spike
            if (close[i] > r1_aligned[i] and 
                uptrend_4h_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with 4h downtrend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  downtrend_4h_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price closes below S1 OR 4h trend changes to downtrend
            if (close[i] < s1_aligned[i] or not uptrend_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price closes above R1 OR 4h trend changes to uptrend
            if (close[i] > r1_aligned[i] or not downtrend_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0