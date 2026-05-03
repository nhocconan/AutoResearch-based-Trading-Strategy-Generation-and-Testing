#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels identify high-probability intraday support/resistance.
# Breakout above R3 or below S3 with 1d EMA34 trend alignment captures strong momentum moves.
# Volume spike confirms institutional participation. Designed for 12-25 trades/year on 12h
# to minimize fee drag while maintaining edge in both bull and bear markets via trend filter.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Camarilla levels using prior 12h bar
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous bar for pivot calc
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous 12h bar
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Avoid division by zero
        if prev_high == prev_low:
            continue
            
        # Camarilla R3 and S3 levels
        camarilla_range = prev_high - prev_low
        r3 = prev_close + camarilla_range * 1.1 / 4
        s3 = prev_close - camarilla_range * 1.1 / 4
        
        # Volume confirmation: 24-period EMA (2 days of 12h bars)
        vol_ema_24 = pd.Series(volume[max(0, i-23):i+1]).ewm(span=24, adjust=False, min_periods=1).mean().iloc[-1] if i >= 23 else volume[i]
        volume_spike = volume[i] > (2.0 * vol_ema_24)
        
        if position == 0:
            # Long: break above R3 in 1d uptrend with volume spike
            if close[i] > r3 and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 in 1d downtrend with volume spike
            elif close[i] < s3 and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 or loses 1d uptrend
            if close[i] < r3 or ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 or loses 1d downtrend
            if close[i] > s3 or ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals