#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with volume confirmation and ADX trend filter
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels
# work in both bull and bear markets when combined with trend filter
# Volume > 1.3x average confirms reversal strength
# ADX > 20 ensures we only trade when trend exists (avoids choppy markets)
# Target: 25-35 trades/year per symbol to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams %R (14 periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    wr = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)  # avoid division by zero
    
    # Calculate 1d ADX (14 periods) for trend filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align Williams %R and ADX to 4h timeframe
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 20 indicates trending market (not choppy)
        trending = adx_aligned[i] > 20
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 from oversold + volume + trend
            if (wr_aligned[i] > -80 and wr_aligned[i-1] <= -80 and 
                volume_confirmed and 
                trending):
                position = 1
                signals[i] = position_size
            # Enter short: Williams %R crosses below -20 from overbought + volume + trend
            elif (wr_aligned[i] < -20 and wr_aligned[i-1] >= -20 and 
                  volume_confirmed and 
                  trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion) or crosses below -80
            if wr_aligned[i] >= -50 or wr_aligned[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion) or crosses above -20
            if wr_aligned[i] <= -50 or wr_aligned[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsR_MeanReversion_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0