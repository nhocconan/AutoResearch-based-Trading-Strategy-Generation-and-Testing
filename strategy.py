#!/usr/bin/env python3
"""
6h Williams %R + 1d SuperTrend + Volume Spike
Hypothesis: Williams %R identifies oversold/overbought conditions on 6h, while 1d SuperTrend provides the trend direction.
Long when Williams %R < -80 (oversold) AND price above SuperTrend AND volume spike.
Short when Williams %R > -20 (overbought) AND price below SuperTrend AND volume spike.
This mean-reversion-with-trend approach works in both bull and bear markets by only taking trades in the direction of the higher timeframe trend.
Volume spike confirms institutional participation. Target: 12-37 trades/year.
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
    
    # Get 1d data for SuperTrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # Need sufficient data for SuperTrend
        return np.zeros(n)
    
    # Calculate 1d SuperTrend (ATR=10, multiplier=3.0)
    if len(df_1d) >= 10:
        # True Range
        tr1 = pd.Series(df_1d['high']).diff().abs()
        tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
        tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=10, min_periods=10).mean().values
        
        # SuperTrend calculation
        hl2 = (pd.Series(df_1d['high']) + pd.Series(df_1d['low'])) / 2
        upperband = hl2 + (3.0 * atr)
        lowerband = hl2 - (3.0 * atr)
        
        # Initialize SuperTrend
        supertrend = np.zeros_like(hl2)
        direction = np.ones_like(hl2)  # 1 for uptrend, -1 for downtrend
        
        supertrend[0] = hl2.iloc[0]
        direction[0] = 1
        
        for i in range(1, len(hl2)):
            if i < 10:  # Not enough data for ATR yet
                supertrend[i] = hl2.iloc[i]
                direction[i] = 1
                continue
                
            # Upper and lower band logic
            if upperband.iloc[i] < supertrend[i-1] or close_1d.iloc[i-1] > supertrend[i-1]:
                upperband.iloc[i] = hl2.iloc[i] + (3.0 * atr[i])
            
            if lowerband.iloc[i] > supertrend[i-1] or close_1d.iloc[i-1] < supertrend[i-1]:
                lowerband.iloc[i] = hl2.iloc[i] - (3.0 * atr[i])
            
            # Trend direction
            if close_1d.iloc[i] > supertrend[i-1]:
                direction[i] = 1
            elif close_1d.iloc[i] < supertrend[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
            
            # SuperTrend value
            if direction[i] == 1:
                supertrend[i] = lowerband.iloc[i]
            else:
                supertrend[i] = upperband.iloc[i]
        
        # Align SuperTrend and direction to 6h timeframe
        supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
        direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    else:
        supertrend_aligned = np.full(n, close[0])  # fallback
        direction_aligned = np.full(n, 1)  # default uptrend
    
    # Calculate 6h Williams %R (14-period)
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / np.where((highest_high - lowest_low) != 0, (highest_high - lowest_low), 1)
    else:
        williams_r = np.full(n, -50)  # neutral
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        wr_value = williams_r[i]
        supertrend_value = supertrend_aligned[i]
        trend_direction = direction_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price above SuperTrend AND uptrend AND volume spike
            long_condition = (wr_value < -80) and (curr_close > supertrend_value) and (trend_direction == 1) and volume_spike
            # Short: Williams %R overbought (> -20) AND price below SuperTrend AND downtrend AND volume spike
            short_condition = (wr_value > -20) and (curr_close < supertrend_value) and (trend_direction == -1) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: Williams %R returns above -50 or price closes below SuperTrend
            if wr_value > -50 or curr_close <= supertrend_value:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 or price closes above SuperTrend
            if wr_value < -50 or curr_close >= supertrend_value:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_SuperTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0