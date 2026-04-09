#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ADX regime filter
# - Long when price breaks above Donchian upper channel (20-period high) + volume spike + 1d ADX > 25 (trending)
# - Short when price breaks below Donchian lower channel (20-period low) + volume spike + 1d ADX > 25 (trending)
# - Exit when price returns to Donchian midpoint (mean reversion) or ADX < 20 (range regime)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years) to avoid fee drag
# - Combines breakout momentum with regime filtering for robustness in bull/bear markets

name = "4h_1d_donchian_breakout_volume_adx_v1"
timeframe = "4h"
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
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.where(tr_smooth != 0, tr_smooth, np.nan)
    minus_di = 100 * minus_dm_smooth / np.where(tr_smooth != 0, tr_smooth, np.nan)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.nan)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 4h Donchian Channel (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion or regime change
            if close[i] <= donchian_mid[i]:  # Return to midpoint
                position = 0
                signals[i] = 0.0
            elif adx_1d_aligned[i] < 20:  # Range regime
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion or regime change
            if close[i] >= donchian_mid[i]:  # Return to midpoint
                position = 0
                signals[i] = 0.0
            elif adx_1d_aligned[i] < 20:  # Range regime
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries with volume confirmation and trending regime
            if (close[i] > donchian_upper[i] and  # Break above upper channel
                volume_spike[i] and              # Volume confirmation
                adx_1d_aligned[i] > 25):         # Trending regime (ADX > 25)
                position = 1
                signals[i] = 0.25
            elif (close[i] < donchian_lower[i] and  # Break below lower channel
                  volume_spike[i] and               # Volume confirmation
                  adx_1d_aligned[i] > 25):          # Trending regime (ADX > 25)
                position = -1
                signals[i] = -0.25
    
    return signals