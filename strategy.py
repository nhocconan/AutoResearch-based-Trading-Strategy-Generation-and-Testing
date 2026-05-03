#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 12h EMA50 Trend + Volume Spike
# Williams %R identifies overbought/oversold conditions. Extreme readings (< -90 or > -10)
# combined with 12h EMA50 trend filter and volume spike provide high-conviction entries
# in both bull and bear markets. Designed for 12-30 trades/year on 6h to minimize fee drag
# while capturing strong reversals and continuations.

name = "6h_WilliamsR_Extreme_12hEMA50_VolumeSpike"
timeframe = "6h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R (14-period) on primary timeframe
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Williams %R using data up to current bar
        lookback = min(14, i+1)
        highest_high = np.max(high[i-lookback+1:i+1])
        lowest_low = np.min(low[i-lookback+1:i+1])
        
        # Avoid division by zero
        if highest_high == lowest_low:
            williams_r = -50  # neutral value
        else:
            williams_r = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        
        # Volume confirmation: current volume > 2.0 * 20-period EMA of volume
        vol_lookback = min(20, i+1)
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (2.0 * vol_ema_20)
        
        # Williams %R extreme conditions
        williams_oversold = williams_r < -90  # extreme oversold
        williams_overbought = williams_r > -10  # extreme overbought
        
        if position == 0:
            # Long: Williams %R extreme oversold + 12h uptrend + volume spike
            if williams_oversold and ema_50_12h_aligned[i] > close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R extreme overbought + 12h downtrend + volume spike
            elif williams_overbought and ema_50_12h_aligned[i] < close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns from extreme or loses 12h uptrend
            if williams_r > -50 or ema_50_12h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns from extreme or loses 12h downtrend
            if williams_r < -50 or ema_50_12h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals