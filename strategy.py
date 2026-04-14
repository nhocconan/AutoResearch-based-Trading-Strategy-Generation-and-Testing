#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + volume + ADX trend filter
# Williams %R < -80 (oversold) in uptrend (ADX > 25) for long entries
# Williams %R > -20 (overbought) in downtrend (ADX > 25) for short entries
# Volume confirmation: current volume > 1.5x 20-period average
# Works in both bull and bear markets: trend filter prevents counter-trend trades,
# Williams %R captures mean reversion within trend
# Uses 4h Williams %R for timing, 4h ADX for trend regime

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14 periods)
    williams_len = 14
    highest_high = pd.Series(high).rolling(window=williams_len, min_periods=williams_len).max().values
    lowest_low = pd.Series(low).rolling(window=williams_len, min_periods=williams_len).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate ADX (14 periods)
    adx_len = 14
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, williams_len, adx_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx[i] > 25
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Williams %R signals
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        if position == 0:
            # Enter long: trending + oversold + volume confirmation
            if trending and oversold and volume_confirm:
                position = 1
                signals[i] = position_size
            # Enter short: trending + overbought + volume confirmation
            elif trending and overbought and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (momentum fading) OR ADX drops
            if williams_r[i] > -50 or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (momentum fading) OR ADX drops
            if williams_r[i] < -50 or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsR_Volume_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0