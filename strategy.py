#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels from daily timeframe provide high-probability breakout points.
# Trading breakouts of R3 (resistance 3) or S3 (support 3) with 1d EMA34 trend filter
# and volume confirmation captures strong moves while minimizing false signals.
# Designed for 20-50 trades/year on 4h to avoid fee drag and work in both bull/bear markets.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels: based on previous day's OHLC
    # R3 = close + ((high-low)*1.1/4)
    # S3 = close - ((high-low)*1.1/4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = (high_1d - low_1d) * 1.1
    r3 = close_1d + camarilla_range / 4
    s3 = close_1d - camarilla_range / 4
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 4h data for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks R3 or S3 with volume spike
        breakout_long = close[i] > r3_aligned[i] and volume_spike[i]
        breakout_short = close[i] < s3_aligned[i] and volume_spike[i]
        
        if position == 0:
            # Long: break above R3 in 1d uptrend with volume spike
            if breakout_long and ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 in 1d downtrend with volume spike
            elif breakout_short and ema_34_1d_aligned[i] < close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below R3 or loses 1d uptrend
            if close[i] < r3_aligned[i] or ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above S3 or loses 1d downtrend
            if close[i] > s3_aligned[i] or ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals