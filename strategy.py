#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Extreme with 1w EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. Extreme readings (< -90 for long, > -10 for short)
# combined with 1w EMA50 trend filter capture mean-reversion in strong trends with controlled frequency.
# Volume spike confirms conviction. Designed for 12-37 trades/year on 12h to minimize fee drag.
# Works in both bull and bear markets by fading extremes in the direction of the higher timeframe trend.

name = "12h_WilliamsR_Extreme_1wEMA50_VolumeSpike"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R (14-period)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after sufficient warmup for Williams %R
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or not in_session[i]):
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
            williams_r = -50.0
        else:
            williams_r = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        
        # Volume confirmation: 20-period EMA
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Extreme Williams %R conditions
        williams_r_oversold = williams_r < -90  # Extreme oversold
        williams_r_overbought = williams_r > -10  # Extreme overbought
        
        if position == 0:
            # Long: extreme oversold in 1w uptrend with volume spike
            if williams_r_oversold and ema_50_1w_aligned[i] > close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: extreme overbought in 1w downtrend with volume spike
            elif williams_r_overbought and ema_50_1w_aligned[i] < close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns from extreme or loses 1w uptrend
            if williams_r > -50 or ema_50_1w_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns from extreme or loses 1w downtrend
            if williams_r < -50 or ema_50_1w_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals