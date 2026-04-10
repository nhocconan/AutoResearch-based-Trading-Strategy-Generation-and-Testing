#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# - Long: price breaks above Donchian(20) high + 1d volume > 1.3x 20-period MA + chop < 61.8 (trending)
# - Short: price breaks below Donchian(20) low + 1d volume > 1.3x 20-period MA + chop < 61.8 (trending)
# - Exit: price returns to Donchian(20) midline (mean of 20-period high/low)
# - Position sizing: 0.30 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year)
# - Donchian breakouts capture strong trending moves
# - Volume confirmation ensures institutional participation
# - Chop filter avoids false breakouts in ranging markets

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d ADX(14) for chop regime filter (using ADX < 25 as choppy)
    # ADX requires +DI, -DI, and TR
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di_1d = 100 * (pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d)
    minus_di_1d = 100 * (pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d)
    dx_1d = 100 * abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Handle division by zero in ADX calculation
    adx_1d = np.where((plus_di_1d + minus_di_1d) == 0, 0, adx_1d)
    
    # Chop regime: ADX < 25 indicates choppy/ranging market (avoid breakouts here)
    chop_regime = adx_1d < 25  # True when choppy, False when trending
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d data for current 12h bar (completed 1d bar)
        chop_current = chop_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1d_current = volume_1d_aligned[i]
        
        # Regime condition: NOT choppy (trending market) = ADX >= 25
        trending_regime = not chop_current
        
        # Volume spike condition: current 1d volume > 1.3x 20-period MA
        volume_spike = volume_1d_current > 1.3 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + trending regime + volume spike
            if (close_12h[i] > donchian_high[i] and trending_regime and volume_spike):
                position = 1
                signals[i] = 0.30
            # Short entry: price breaks below Donchian low + trending regime + volume spike
            elif (close_12h[i] < donchian_low[i] and trending_regime and volume_spike):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to Donchian midline
            if position == 1:
                if close_12h[i] <= donchian_mid[i]:  # Exit long when price <= midline
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                if close_12h[i] >= donchian_mid[i]:  # Exit short when price >= midline
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals