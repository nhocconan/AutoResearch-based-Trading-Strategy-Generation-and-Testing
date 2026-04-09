#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike confirmation
# - Uses 6h Williams %R(14) for mean reversion signals: long when %R < -80 (oversold), short when %R > -20 (overbought)
# - Filters with 1d ADX(14) > 25 to ensure we trade only in trending markets (avoid chop)
# - Confirms with 6h volume > 1.8x 20-period average for institutional participation
# - Exits when Williams %R reverts to -50 (mean reversion target) or opposite extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years) to minimize fee drag
# - Williams %R is effective in both bull and bear markets for catching reversals at extremes
# - ADX filter ensures we avoid false signals in ranging markets
# - Volume confirmation adds reliability to breakouts from extreme levels

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
    
    # Smoothed DM and TR for ADX
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    plus_di = np.where(atr_1d > 0, (plus_dm_smooth / atr_1d) * 100, 0)
    minus_di = np.where(atr_1d > 0, (minus_dm_smooth / atr_1d) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 1d ADX > 25 (trending market)
    adx_trending = adx_1d > 25
    
    # Align 1d ADX to 6h
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending.astype(float))
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # 6h Volume > 1.8x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(adx_trending_aligned[i]) or
            np.isnan(volume_spike[i]) or adx_trending_aligned[i] < 1):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Williams %R reverts to -50 or reaches overbought
            if williams_r[i] >= -50:  # Mean reversion target reached
                position = 0
                signals[i] = 0.0
            elif williams_r[i] > -20:  # Overbought - exit long
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R reverts to -50 or reaches oversold
            if williams_r[i] <= -50:  # Mean reversion target reached
                position = 0
                signals[i] = 0.0
            elif williams_r[i] < -80:  # Oversold - exit short
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with ADX filter and volume confirmation
            if (williams_r[i] < -80 and      # Oversold
                adx_trending_aligned[i] and  # Trending market (ADX > 25)
                volume_spike[i]):            # Volume confirmation
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] > -20 and    # Overbought
                  adx_trending_aligned[i] and # Trending market (ADX > 25)
                  volume_spike[i]):           # Volume confirmation
                position = -1
                signals[i] = -0.25
    
    return signals