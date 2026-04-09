#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + ADX regime filter with volume confirmation
# - Williams %R(14) on 6h for overbought/oversold signals (%R < -80 for long, > -20 for short)
# - ADX(14) on 1d for regime filter: only trade when ADX > 25 (trending market)
# - Volume confirmation: current 6h volume > 1.5x 20-period average volume
# - Only takes long signals in uptrend (ADX rising) and short signals in downtrend
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years) to avoid fee drag
# - Williams %R is effective in catching reversals in trending markets, which suits 6h timeframe
# - ADX regime filter prevents choppy market losses, volume confirmation ensures conviction

name = "6h_1d_williams_adx_volume_v1"
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
    
    # Pre-compute 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1_1d[0] if len(tr1_1d) > 0 else 0
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR and DM
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.divide(dm_plus_14, tr_14, out=np.zeros_like(dm_plus_14), where=tr_14!=0) * 100
    di_minus = np.divide(dm_minus_14, tr_14, out=np.zeros_like(dm_minus_14), where=tr_14!=0) * 100
    
    # DX and ADX
    dx = np.divide(np.abs(di_plus - di_minus), (di_plus + di_minus), 
                   out=np.zeros_like(di_plus), where=(di_plus + di_minus)!=0) * 100
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.divide((highest_high - close), (highest_high - lowest_low), 
                           out=np.full_like(close, -50.0), where=(highest_high - lowest_low)!=0) * -100
    
    # 6h volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        if adx_1d_aligned[i] <= 25:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R returns from oversold or ADX weakens
            if williams_r[i] > -50:  # Return from oversold
                position = 0
                signals[i] = 0.0
            elif adx_1d_aligned[i] < 20:  # Trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns from overbought or ADX weakens
            if williams_r[i] < -50:  # Return from overbought
                position = 0
                signals[i] = 0.0
            elif adx_1d_aligned[i] < 20:  # Trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entries aligned with trend
            if (williams_r[i] < -80 and  # Oversold
                volume_confirm[i] and    # Volume confirmation
                adx_1d_aligned[i] > 25): # Trending market
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] > -20 and  # Overbought
                  volume_confirm[i] and    # Volume confirmation
                  adx_1d_aligned[i] > 25): # Trending market
                position = -1
                signals[i] = -0.25
    
    return signals