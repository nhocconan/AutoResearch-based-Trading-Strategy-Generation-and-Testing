#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 12h EMA trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries.
# 12h EMA ensures we trade with the intermediate trend (long in uptrend, short in downtrend).
# Volume spike confirms institutional participation. Designed for 15-30 trades/year on 4h to minimize fee drag.
# Works in bull markets via buying pullbacks in uptrends and in bear markets via selling rallies in downtrends.

name = "4h_WilliamsR_12hEMA_VolumeSpike"
timeframe = "4h"
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
    
    # Calculate Williams %R(14) on 4h data
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after sufficient warmup for Williams %R
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Williams %R using data up to current bar
        highest_high = np.max(high[i-13:i+1])  # 14-period high
        lowest_low = np.min(low[i-13:i+1])     # 14-period low
        
        # Avoid division by zero
        if highest_high == lowest_low:
            williams_r = -50.0  # neutral value
        else:
            williams_r = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        
        # Volume confirmation: 20-period EMA
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Williams %R conditions
        oversold = williams_r < -80
        overbought = williams_r > -20
        
        if position == 0:
            # Long: oversold in 12h uptrend with volume spike
            if oversold and ema_50_12h_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: overbought in 12h downtrend with volume spike
            elif overbought and ema_50_12h_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral or loses 12h uptrend
            if williams_r > -50 or ema_50_12h_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral or loses 12h downtrend
            if williams_r < -50 or ema_50_12h_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals