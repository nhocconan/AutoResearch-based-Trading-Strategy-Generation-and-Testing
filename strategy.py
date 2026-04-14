#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume confirmation
# Williams %R(14) identifies overbought/oversold conditions for mean reversion
# 1d ADX(14) > 25 filters for trending markets where mean reversion is less effective
# Volume > 1.5x 20-period EMA confirms institutional participation during reversals
# Target: 15-35 trades/year with mean reversion logic suited for 2025 range-bound conditions
# Exits via opposite Williams %R threshold to avoid whipsaws

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 1d ADX (14-period) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ADX
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(14, n):
        # Get aligned 1d ADX
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        if np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned) or \
           np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        # Trend filter: ADX > 25 indicates trending market (less effective for mean reversion)
        # We want ranging markets: ADX < 25
        ranging = adx_1d_aligned < 25
        
        if position == 0:  # No position - look for mean reversion entries
            # Long: Williams %R oversold (< -80) in ranging market with volume
            if williams_r[i] < -80 and volume_confirm and ranging:
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought (> -20) in ranging market with volume
            elif williams_r[i] > -20 and volume_confirm and ranging:
                position = -1
                signals[i] = -position_size
        elif position == 1:  # Long position - exit when Williams %R returns from oversold
            # Exit if Williams %R rises above -50 (returning from oversold)
            if williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when Williams %R returns from overbought
            # Exit if Williams %R falls below -50 (returning from overbought)
            if williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsR_1dADX_MeanRev"
timeframe = "12h"
leverage = 1.0