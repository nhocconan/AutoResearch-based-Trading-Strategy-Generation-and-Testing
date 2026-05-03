#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout + 4h EMA50 trend filter + volume confirmation
# Camarilla pivots provide intraday support/resistance levels that work in ranging and trending markets.
# 4h EMA > 50 ensures medium-term trend alignment to reduce whipsaws.
# Volume confirmation (1.8x 20-period EMA) filters low-momentum breakouts.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.
# Discrete sizing (0.20) minimizes fee churn. Target: 60-120 total trades over 4 years.

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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate Camarilla levels from previous 1h bar
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    camarilla_r3 = (close_series.shift(1) + (high_series.shift(1) - low_series.shift(1)) * 1.1 / 2).values
    camarilla_s3 = (close_series.shift(1) - (high_series.shift(1) - low_series.shift(1)) * 1.1 / 2).values
    
    # Volume confirmation: 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have valid Camarilla levels
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Uptrend: close > EMA50, Downtrend: close < EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 in uptrend with volume spike
            if close[i] > camarilla_r3[i] and uptrend and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 in downtrend with volume spike
            elif close[i] < camarilla_s3[i] and downtrend and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or loses uptrend
            if close[i] < camarilla_s3[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R3 or loses downtrend
            if close[i] > camarilla_r3[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals