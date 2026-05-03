#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 1h timeframe for entry timing, 4h for trend direction and Camarilla levels, daily for volume baseline.
# Camarilla R1/S1 provides tight intraday support/resistance for precise entries.
# 4h EMA50 ensures we only trade with the intermediate trend to avoid whipsaws.
# Volume spike (2x 20-period EMA) confirms institutional participation.
# Designed for 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.
# Position size fixed at 0.20 to manage drawdown in bear markets like 2022.

name = "1h_Camarilla_R1S1_4hEMA50_VolumeSpike"
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
    
    # Get 4h data for EMA50 trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    # Camarilla: R1 = close + 0.1083*(high-low), S1 = close - 0.1083*(high-low)
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    
    R1 = prev_close_4h + 0.1083 * (prev_high_4h - prev_low_4h)
    S1 = prev_close_4h - 0.1083 * (prev_high_4h - prev_low_4h)
    
    # Align Camarilla levels to 1h timeframe (use previous 4h bar's levels)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # Get 1d data for volume baseline (more stable than 4h volume EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume EMA20 for volume spike filter
    vol_ema_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ema_20_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20_1d_aligned[i])
        
        if position == 0:
            # Long: price breaks above R1 in uptrend alignment with volume spike
            if close[i] > R1_aligned[i] and ema_50_4h_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 in downtrend alignment with volume spike
            elif close[i] < S1_aligned[i] and ema_50_4h_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 or loses uptrend alignment
            if close[i] < S1_aligned[i] or ema_50_4h_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R1 or loses downtrend alignment
            if close[i] > R1_aligned[i] or ema_50_4h_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals