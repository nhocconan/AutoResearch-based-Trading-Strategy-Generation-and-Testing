#!/usr/bin/env python3
"""
12h Williams %R Mean Reversion + 1d EMA34 Trend Filter + Volume Spike
Hypothesis: Williams %R identifies overbought/oversold conditions. In trending markets (price > 1d EMA34 for long, < for short),
extreme %R readings offer high-probability mean-reversion entries. Volume spike confirms institutional participation.
Works in bull/bear via trend filter. Target: 12-37 trades/year (50-150 over 4 years).
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
    
    # Get 1d data for EMA34 trend filter and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.divide((highest_high - df_1d['close'].values), denominator, out=np.full_like(denominator, -50.0), where=denominator!=0) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1d indicators (34 for EMA, 14 for Williams %R)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(williams_r_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        williams_r_val = williams_r_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA34 (uptrend) AND volume spike
            long_condition = (williams_r_val < -80) and (curr_close > ema_trend) and volume_spike
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA34 (downtrend) AND volume spike
            short_condition = (williams_r_val > -20) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion) or trend breaks
            if williams_r_val >= -50 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 or trend breaks
            if williams_r_val <= -50 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_MeanReversion_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0