#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d ADX Regime + Volume Spike
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Williams %R identifies overextended moves: < -80 = oversold, > -20 = overbought
# 1d ADX > 25 indicates trending market (fade extremes), ADX < 20 indicates ranging (mean revert)
# Volume spike (2x 20-period average) confirms institutional participation at extremes
# Works in bull markets via mean reversion from overextension in trends
# Works in bear markets via fading false breakouts during ranging periods
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

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
    
    # Calculate 1d ADX regime (prior completed 1d bar's ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # True Range for ADX
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'] - np.roll(df_1d['high'], 1)) > (np.roll(df_1d['low'], 1) - df_1d['low']),
                       np.maximum(df_1d['high'] - np.roll(df_1d['high'], 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d['low'], 1) - df_1d['low']) > (df_1d['high'] - np.roll(df_1d['high'], 1)),
                        np.maximum(np.roll(df_1d['low'], 1) - df_1d['low'], 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(atr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_smooth
    di_minus = 100 * dm_minus_smooth / atr_smooth
    
    # ADX calculation
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ADX regime: >25 = trending, <20 = ranging
    adx_regime_trending = adx > 25
    adx_regime_ranging = adx < 20
    
    # Align to 6h timeframe (wait for completed 1d bar)
    adx_regime_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_regime_trending)
    adx_regime_ranging_aligned = align_htf_to_ltf(prices, df_1d, adx_regime_ranging)
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 6h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(adx_regime_trending_aligned[i]) or 
            np.isnan(adx_regime_ranging_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) AND volume spike AND 
            # (ADX trending AND price > close 10 bars ago) OR (ADX ranging)
            if (williams_r[i] < -80 and 
                volume_spike[i] and 
                ((adx_regime_trending_aligned[i] and close[i] > close[i-10]) or 
                 adx_regime_ranging_aligned[i])):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought) AND volume spike AND 
            # (ADX trending AND price < close 10 bars ago) OR (ADX ranging)
            elif (williams_r[i] > -20 and 
                  volume_spike[i] and 
                  ((adx_regime_trending_aligned[i] and close[i] < close[i-10]) or 
                   adx_regime_ranging_aligned[i])):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -50 (return from oversold) OR Williams %R > -20 (overbought)
            if williams_r[i] > -50 or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -50 (return from overbought) OR Williams %R < -80 (oversold)
            if williams_r[i] < -50 or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals