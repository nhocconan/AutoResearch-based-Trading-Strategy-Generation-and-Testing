#!/usr/bin/env python3
"""
1d Williams %R Mean Reversion with Weekly EMA Trend Filter and Volume Spike
Hypothesis: Williams %R identifies oversold/overbought conditions. In strong weekly trends (above/below 50 EMA), 
these extreme readings often precede continuation moves rather than reversals. Volume spike confirms institutional 
participation. Works in both bull/bear markets by aligning with weekly trend direction.
Target: 7-25 trades/year (30-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    williams_r = np.divide((highest_high - close), hl_range, out=np.full_like(hl_range, -50.0), where=hl_range!=0) * -100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Williams %R (14) + EMA50 warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        williams_r_val = williams_r[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Williams %R signals with weekly trend filter
        if position == 0:
            # Long: Williams %R below -80 (oversold) AND price above weekly EMA50 (uptrend) AND volume spike
            long_condition = (williams_r_val < -80) and (curr_close > ema_trend) and volume_spike
            # Short: Williams %R above -20 (overbought) AND price below weekly EMA50 (downtrend) AND volume spike
            short_condition = (williams_r_val > -20) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 (momentum fading) OR trend breaks
            if williams_r_val > -50 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 (momentum fading) OR trend breaks
            if williams_r_val < -50 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_MeanReversion_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0