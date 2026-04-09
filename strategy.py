#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX regime filter
# - Uses 6h Williams %R(14) for overbought/oversold conditions
# - Uses 1d ADX(14) to filter ranging vs trending markets
# - Enters long when Williams %R < -80 (oversold) and 1d ADX < 25 (ranging/weak trend)
# - Enters short when Williams %R > -20 (overbought) and 1d ADX < 25
# - Exits when Williams %R returns to -50 (mean reversion center)
# - In strong 1d trends (ADX >= 25), we stay flat to avoid trend-following whipsaws
# - Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years)
# - Williams %R is effective at identifying reversals in ranging markets
# - ADX filter prevents trading during strong trends where mean reversion fails

name = "6h_1d_williamsr_adx_meanrev_v1"
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
    
    # Pre-compute session filter (08-20 UTC) - optional for 6h
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d ADX(14) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    atr_smooth = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI and DX
    plus_di = np.divide(plus_dm_smooth, atr_smooth, out=np.zeros_like(plus_dm_smooth), where=atr_smooth!=0) * 100
    minus_di = np.divide(minus_dm_smooth, atr_smooth, out=np.zeros_like(minus_dm_smooth), where=atr_smooth!=0) * 100
    dx = np.divide(np.abs(plus_di - minus_di), (plus_di + minus_di), out=np.zeros_like(plus_di), where=(plus_di + minus_di)!=0) * 100
    
    # ADX is smoothed DX
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.divide(
        (highest_high - close), 
        (highest_high - lowest_low), 
        out=np.zeros_like(highest_high), 
        where=(highest_high - lowest_low)!=0
    ) * -100
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            (highest_high[i] - lowest_low[i]) == 0):
            signals[i] = 0.0
            continue
        
        # Only trade in ranging/weak trend markets (ADX < 25)
        if adx_1d_aligned[i] >= 25:
            # Strong trend - stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Williams %R returns to mean reversion level
            if williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R returns to mean reversion level
            if williams_r[i] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries
            if williams_r[i] < -80:  # Oversold
                position = 1
                signals[i] = 0.25
            elif williams_r[i] > -20:  # Overbought
                position = -1
                signals[i] = -0.25
    
    return signals