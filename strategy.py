#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivots identify intraday support/resistance levels. Breakouts above R3 or below S3
# with volume spike capture strong intraday moves. 4h EMA50 ensures alignment with higher timeframe trend.
# Session filter (08-20 UTC) reduces noise. Designed for 15-37 trades/year on 1h to minimize fee drag.

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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate daily Camarilla levels (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_range = prior_high - prior_low
    r3 = prior_close + camarilla_range * 1.1 / 4
    s3 = prior_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have prior day data
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: volume > 1.5 * 20-period EMA
        if i >= 20:
            vol_ema = pd.Series[volume[max(0, i-19):i+1]].ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema)
        
        if position == 0:
            # Long: break above R3 in 4h uptrend with volume spike
            if close[i] > r3_aligned[i] and ema_50_4h_aligned[i] > close[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: break below S3 in 4h downtrend with volume spike
            elif close[i] < s3_aligned[i] and ema_50_4h_aligned[i] < close[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below R3 or loses 4h uptrend
            if close[i] < r3_aligned[i] or ema_50_4h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above S3 or loses 4h downtrend
            if close[i] > s3_aligned[i] or ema_50_4h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals