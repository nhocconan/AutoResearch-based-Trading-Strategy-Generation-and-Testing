#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout (R3/S3) + 4h volume spike + session filter (08-20 UTC)
# Uses 4h trend direction via EMA50 to avoid counter-trend trades. Volume spike confirms institutional interest.
# Session filter reduces noise during low-liquidity hours. Designed for low trade frequency (15-37/year) to minimize fee drag.
# Works in bull/bear markets by aligning with 4h trend and requiring volume confirmation.

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
    
    # Get 4h data for trend and volume spike
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_4h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_4h['volume'].values > (2.0 * vol_ema_20)
    
    # Align 4h indicators to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike)
    
    # Calculate 1h Camarilla levels (R3, S3, R4, S4) based on previous day's OHLC
    # We need daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC (Camarilla uses prior day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3, S3, R4, S4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align daily Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 4h trend filter: only trade in direction of 4h EMA50
        uptrend = close > ema_50_aligned[i]
        downtrend = close < ema_50_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume spike + 4h uptrend
            if high[i] > r3_aligned[i] and volume_spike_aligned[i] and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 + volume spike + 4h downtrend
            elif low[i] < s3_aligned[i] and volume_spike_aligned[i] and downtrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (reversal) or hits R4 (profit target)
            if low[i] < s3_aligned[i] or high[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R3 (reversal) or hits S4 (profit target)
            if high[i] > r3_aligned[i] or low[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals