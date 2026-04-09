#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX trend filter + volume confirmation
# - Uses 6h Williams %R(14) for overbought/oversold signals (long < -80, short > -20)
# - Confirms with 1d ADX(14) > 25 (strong trend) to avoid choppy markets
# - Adds 1d volume > 1.5x 20-period average for institutional participation
# - Exits when Williams %R reverts to -50 (mean reversion in trend)
# - Position size: 0.25 (25% of capital) for balanced risk/return
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years) to minimize fee drag
# - Works in bull markets (trend continuation) and bear markets (trend continuation)
# - Williams %R captures momentum extremes, ADX filters non-trending conditions

name = "6h_1d_williamsr_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d True Range for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ADX components
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # 1d ATR(14) for ADX
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d smoothed +/- DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # 1d DI+ and DI-
    plus_di = np.where(atr_1d > 0, 100 * plus_dm_smooth / atr_1d, 0)
    minus_di = np.where(atr_1d > 0, 100 * minus_dm_smooth / atr_1d, 0)
    
    # 1d DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 1d Volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * avg_volume_20)
    
    # Align all 1d indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or adx_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Williams %R reverts to -50 (mean reversion in trend)
            if williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R reverts to -50 (mean reversion in trend)
            if williams_r[i] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with ADX trend filter and volume confirmation
            if (williams_r[i] < -80 and  # Oversold
                adx_aligned[i] > 25 and      # Strong trend
                volume_spike_aligned[i]):    # Volume confirmation
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] > -20 and   # Overbought
                  adx_aligned[i] > 25 and      # Strong trend
                  volume_spike_aligned[i]):    # Volume confirmation
                position = -1
                signals[i] = -0.25
    
    return signals