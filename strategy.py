#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivots identify key intraday support/resistance levels; breaks of R3/S3 with
# volume spike and alignment to 4h trend provide high-probability continuation trades.
# Uses 4h/1d for signal direction, 1h only for entry timing to minimize trades (target: 15-37/year).
# Session filter (08-20 UTC) reduces noise. Works in bull/bear by trading breakouts in
# direction of higher timeframe trend.

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike"
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
    
    # Get 4h data for trend filter and volume confirmation (HTF for direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_4h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_4h['volume'].values > (2.0 * vol_ema_20)
    
    # Align 4h indicators to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike)
    
    # Calculate daily data for Camarilla pivot points (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R3/S3 = C ± (H-L)*1.1/4
    camarilla_range = (prev_high - prev_low) * 1.1 / 4
    r3 = prev_close + camarilla_range * 3
    s3 = prev_close - camarilla_range * 3
    
    # Align Camarilla levels to 1h timeframe (using prior day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start after first bar to have prior close
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume spike in uptrend (4h EMA50 rising)
            if close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and ema_50_aligned[i] > ema_50_aligned[i-1] and volume_spike_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 with volume spike in downtrend (4h EMA50 falling)
            elif close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and ema_50_aligned[i] < ema_50_aligned[i-1] and volume_spike_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price re-enters below R3
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price re-enters above S3
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals