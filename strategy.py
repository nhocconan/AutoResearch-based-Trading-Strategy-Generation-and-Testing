#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Camarilla pivots from 12h provide precise intraday support/resistance levels.
# Breakouts at R3/S3 with 12h trend alignment and volume spike capture momentum with low false signals.
# Designed for 6h timeframe with tight entry conditions (12-37 trades/year) to minimize fee drag.
# Works in both bull and bear markets by trading with the 12h trend and using volume confirmation.

name = "6h_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for Camarilla pivots, EMA, and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (R3, S3, R4, S4)
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # where C = (H+L+CLOSE)/3 of previous period
    df_12h = df_12h.copy()
    df_12h['typical_price'] = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    df_12h['range'] = df_12h['high'] - df_12h['low']
    
    # Pivot and levels based on previous 12h bar
    df_12h['pivot'] = df_12h['typical_price'].shift(1)
    df_12h['R3'] = df_12h['pivot'] + (df_12h['range'].shift(1) * 1.1 / 4)
    df_12h['S3'] = df_12h['pivot'] - (df_12h['range'].shift(1) * 1.1 / 4)
    df_12h['R4'] = df_12h['pivot'] + (df_12h['range'].shift(1) * 1.1 / 2)
    df_12h['S4'] = df_12h['pivot'] - (df_12h['range'].shift(1) * 1.1 / 2)
    
    # Calculate 12h EMA50 for trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_12h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_12h['volume'].values > (2.0 * vol_ema_20)
    
    # Align 12h indicators to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_12h, df_12h['R3'].values)
    S3_aligned = align_htf_to_ltf(prices, df_12h, df_12h['S3'].values)
    R4_aligned = align_htf_to_ltf(prices, df_12h, df_12h['R4'].values)
    S4_aligned = align_htf_to_ltf(prices, df_12h, df_12h['S4'].values)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 in uptrend with volume spike
            if high[i] > R3_aligned[i] and is_uptrend and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 in downtrend with volume spike
            elif low[i] < S3_aligned[i] and is_downtrend and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 (reversal) or hits R4 (take profit)
            if low[i] < S3_aligned[i] or high[i] > R4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R3 (reversal) or hits S4 (take profit)
            if high[i] > R3_aligned[i] or low[i] < S4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals