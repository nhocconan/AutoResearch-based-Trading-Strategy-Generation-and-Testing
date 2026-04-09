#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d ADX trend filter and volume confirmation
# - Williams %R(14) identifies overbought/oversold conditions for mean reversion entries
# - 1d ADX(14) > 25 filters for trending markets (only trade pullbacks in strong trends)
# - Volume > 1.5x 20-period average confirms momentum behind the pullback
# - Designed to work in BOTH bull markets (buy pullbacks in uptrends) and bear markets (sell rallies in downtrends)
# - Target: ~20-30 trades/year (80-120 total over 4 years) to minimize fee drag
# - Williams %R provides timely mean reversion signals, ADX avoids choppy markets, volume improves signal quality

name = "6h_1d_williamsr_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) for trend strength filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_1d = 100 * plus_dm_smooth / (atr_1d + 1e-10)
    minus_di_1d = 100 * minus_dm_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # 6h volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(volume_spike[i]) or
            adx_1d_aligned[i] < 25):  # Require trending market
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Williams %R returns above oversold (mean reversion complete)
            if williams_r[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R returns below overbought (mean reversion complete)
            if williams_r[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries in direction of 1d trend
            # Long: Williams %R oversold (< -80) AND volume spike in uptrend (ADX > 25)
            if (williams_r[i] <= -80 and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: Williams %R overbought (>= -20) AND volume spike in downtrend (ADX > 25)
            elif (williams_r[i] >= -20 and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals