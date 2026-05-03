#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike
# Camarilla R3/S3 levels provide high-probability reversal/breakout zones from 4h price action.
# 4h EMA50 filter ensures we only trade in the direction of the 4h trend.
# Volume spike confirms institutional participation at these key levels.
# Session filter (08-20 UTC) reduces noise trades.
# Target: 15-37 trades/year (60-150 over 4 years) for 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla levels, EMA, and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous 4h bar's OHLC)
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Avoid look-ahead by using previous bar's data
    diff = prev_high - prev_low
    r3 = prev_close + (diff * 1.1 / 4)
    s3 = prev_close - (diff * 1.1 / 4)
    r4 = prev_close + (diff * 1.1 / 2)
    s4 = prev_close - (diff * 1.1 / 2)
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_4h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_4h['volume'].values > (2.0 * vol_ema_20)
    
    # Align 4h indicators to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    r4_aligned = align_htf_to_ltf(prices, df_4h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_4h, s4)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 in uptrend with volume spike
            # OR price breaks above R4 (strong breakout) regardless of trend
            if ((high[i] > r3_aligned[i] and is_uptrend and volume_spike_aligned[i]) or
                (high[i] > r4_aligned[i])):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S3 in downtrend with volume spike
            # OR price breaks below S4 (strong breakout) regardless of trend
            elif ((low[i] < s3_aligned[i] and is_downtrend and volume_spike_aligned[i]) or
                  (low[i] < s4_aligned[i])):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 (reversal) or hits R4 (profit target)
            if low[i] < s3_aligned[i] or high[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price breaks above R3 (reversal) or hits S4 (profit target)
            if high[i] > r3_aligned[i] or low[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals