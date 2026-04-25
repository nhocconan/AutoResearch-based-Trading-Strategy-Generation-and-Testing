#!/usr/bin/env python3
"""
4h Williams %R Reversal + 12h EMA50 Trend + Volume Spike
Hypothesis: Williams %R identifies overbought/oversold conditions. In ranging or choppy markets, 
extreme readings (%R < -80 for oversold, %R > -20 for overbought) often precede mean-reverting 
bounces. Filtering by 12h EMA50 trend ensures we only take reversals aligned with the intermediate 
trend, reducing false signals in strong trends. Volume spike confirms participation. Designed for 
BTC/ETH mean reversion in chop regimes while avoiding excessive trades via strict 12h trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R (14-period) on 4h
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, -100 * (highest_high - close) / denominator, -50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Williams %R and EMA50 warmup
    start_idx = max(lookback, 50)
    
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
        wr = williams_r[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Mean reversion signals with trend filter
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price above 12h EMA50 (uptrend filter)
            long_condition = (wr < -80) and (curr_close > ema_trend) and volume_spike
            # Short: Williams %R overbought (> -20) AND price below 12h EMA50 (downtrend filter)
            short_condition = (wr > -20) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean) or trend breaks
            if wr >= -50 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean) or trend breaks
            if wr <= -50 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_%R_Reversal_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0