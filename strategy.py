#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike
# Camarilla R3/S3 levels from daily price action provide high-probability breakout zones.
# 12h EMA34 filter ensures alignment with the medium-term trend to avoid counter-trend trades.
# Volume spike confirms institutional participation at these key levels.
# Designed for low trade frequency (target: 20-50/year) to minimize fee drag on 4h timeframe.
# Works in both bull and bear markets by trading with the higher timeframe trend.

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34 = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Get 1d data for Camarilla levels and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead by using previous day's data
    diff = prev_high - prev_low
    r3 = prev_close + (diff * 1.1 / 4)
    s3 = prev_close - (diff * 1.1 / 4)
    r4 = prev_close + (diff * 1.1 / 2)
    s4 = prev_close - (diff * 1.1 / 2)
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1d indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend direction
        is_uptrend = close[i] > ema_34_aligned[i]
        is_downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 in uptrend with volume spike
            # OR price breaks above R4 (strong breakout) regardless of trend
            if ((high[i] > r3_aligned[i] and is_uptrend and volume_spike_aligned[i]) or
                (high[i] > r4_aligned[i])):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 in downtrend with volume spike
            # OR price breaks below S4 (strong breakout) regardless of trend
            elif ((low[i] < s3_aligned[i] and is_downtrend and volume_spike_aligned[i]) or
                  (low[i] < s4_aligned[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 (reversal) or hits R4 (profit target)
            if low[i] < s3_aligned[i] or high[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R3 (reversal) or hits S4 (profit target)
            if high[i] > r3_aligned[i] or low[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals