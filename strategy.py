#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13. In strong trends, we fade extreme readings
# in the direction of the 1d EMA34 trend. Volume spike confirms conviction. Designed for 12-30 trades/year
# on 6h to minimize fee drag. Works in both bull and bear markets by fading extremes with higher timeframe trend.

name = "6h_ElderRay_Extreme_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after sufficient warmup for EMA13
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA
        if i >= 19:
            vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Extreme Elder Ray conditions (using 1.5 * std as threshold)
        if i >= 30:  # Need sufficient data for std calculation
            bp_std = np.std(bull_power[max(0, i-29):i+1])
            br_std = np.std(bear_power[max(0, i-29):i+1])
            
            # Avoid division by zero
            if bp_std == 0:
                bp_std = 0.001 * close[i]
            if br_std == 0:
                br_std = 0.001 * close[i]
                
            bull_extreme = bull_power[i] > (1.5 * bp_std)  # Extreme bull power
            bear_extreme = bear_power[i] < (-1.5 * br_std)  # Extreme bear power
        else:
            bull_extreme = False
            bear_extreme = False
        
        if position == 0:
            # Long: extreme bear power (oversold) in 1d uptrend with volume spike
            if bear_extreme and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: extreme bull power (overbought) in 1d downtrend with volume spike
            elif bull_extreme and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bear power normalizes or loses 1d uptrend
            if bear_power[i] < (0.5 * br_std) or ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bull power normalizes or loses 1d downtrend
            if bull_power[i] > (-0.5 * bp_std) or ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals