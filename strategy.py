#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d ADX regime + volume spike
# Williams %R identifies overbought/oversold conditions (below -80 = oversold, above -20 = overbought)
# 1d ADX > 25 indicates trending market (favor continuation), ADX < 20 indicates ranging (favor mean reversion)
# Volume spike (2x 20-period average) confirms institutional participation
# In trending markets (ADX > 25): breakout continuation when %R extremes with volume
# In ranging markets (ADX < 20): mean reversion from %R extremes
# Discrete position sizing: 0.25 balances exposure and risk
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in bull markets via breakout continuation with trend alignment
# Works in bear markets via mean reversion from extremes in ranging conditions

name = "6h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM-
    tr_period = 14
    tr_sum = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=tr_period, min_periods=tr_period).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=tr_period, min_periods=tr_period).sum().values
    
    # DI+ and DI-
    di_plus = np.where(tr_sum != 0, 100 * dm_plus_sum / tr_sum, 0)
    di_minus = np.where(tr_sum != 0, 100 * dm_minus_sum / tr_sum, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Align ADX to 6h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Williams %R
    williams_period = 14
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * (highest_high - close) / (highest_high - lowest_low), -50)
    
    # Calculate 6h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(williams_period, 20, tr_period)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Williams %R extremes
            williams_oversold = williams_r[i] < -80
            williams_overbought = williams_r[i] > -20
            
            if volume_spike[i]:
                # Trending market (ADX > 25): breakout continuation
                if adx_aligned[i] > 25:
                    # Long: oversold breakout continuation
                    if williams_oversold and close[i] > np.maximum(high[i-1], highest_high[i-1]):
                        signals[i] = 0.25
                        position = 1
                    # Short: overbought breakout continuation
                    elif williams_overbought and close[i] < np.minimum(low[i-1], lowest_low[i-1]):
                        signals[i] = -0.25
                        position = -1
                # Ranging market (ADX < 20): mean reversion from extremes
                elif adx_aligned[i] < 20:
                    # Long: oversold mean reversion
                    if williams_oversold:
                        signals[i] = 0.25
                        position = 1
                    # Short: overbought mean reversion
                    elif williams_overbought:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns above -50 (mean reversion) OR ADX drops below 20 (trend weakening)
            if williams_r[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -50 (mean reversion) OR ADX drops below 20 (trend weakening)
            if williams_r[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals