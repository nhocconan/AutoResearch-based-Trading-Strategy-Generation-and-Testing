#!/usr/bin/env python3
"""
6h Williams %R + 1d EMA34 Trend Filter + Volume Spike
Hypothesis: Williams %R identifies overbought/oversold conditions that tend to reverse in ranging markets but continue in trending markets. 
Using 1d EMA34 to filter trend direction: long when Williams %R < -80 (oversold) AND price > 1d EMA34 (uptrend), short when Williams %R > -20 (overbought) AND price < 1d EMA34 (downtrend). 
Volume spike confirms participation. Works in bull/bear by trend-filtering mean-reversion signals.
Target: 12-37 trades/year (50-150 over 4 years).
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.divide(
        (highest_high - close) * -100, 
        denominator, 
        out=np.full_like(close, -50.0), 
        where=denominator!=0
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Williams %R (14) + EMA34 warmup
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        wr = williams_r[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Mean reversion signals with trend filter
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND above 1d EMA34 (uptrend filter)
            long_condition = (wr < -80.0) and (curr_close > ema_trend) and volume_spike
            # Short: Williams %R > -20 (overbought) AND below 1d EMA34 (downtrend filter)
            short_condition = (wr > -20.0) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 or trend breaks
            if wr >= -50.0 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 or trend breaks
            if wr <= -50.0 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0