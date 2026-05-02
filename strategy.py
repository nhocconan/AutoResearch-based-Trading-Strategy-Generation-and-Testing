#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ADX regime filter + volume confirmation
# Donchian breakout provides clear entry/exit levels, ADX filter avoids whipsaw in ranging markets,
# volume confirmation ensures institutional participation. Discrete sizing 0.25 minimizes fee churn.
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits.
# Works in both bull and bear markets by adapting to regime via ADX: trend follow when ADX>25,
# mean revert at Donchian bands when ADX<20 (range). Uses 1d for HTF regime and Donchian calculation.

name = "12h_Donchian20_1dADXRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX regime and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ADX for regime filter (trending vs ranging)
    # True Range
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'].values - np.roll(df_1d['high'].values, 1)) > 
                       (np.roll(df_1d['low'].values, 1) - df_1d['low'].values),
                       np.maximum(df_1d['high'].values - np.roll(df_1d['high'].values, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d['low'].values, 1) - df_1d['low'].values) > 
                        (df_1d['high'].values - np.roll(df_1d['high'].values, 1)),
                        np.maximum(np.roll(df_1d['low'].values, 1) - df_1d['low'].values, 0), 0)
    dm_plus[0] = dm_minus[0] = 0  # first bar
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, ADX and volume MA)
    start_idx = 50  # max(20 for Donchian/volume, 34 for ADX) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from 1d ADX
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # In trending market: follow Donchian breakout
                # Long: price breaks above Donchian high with volume confirmation
                if close[i] > donchian_high_aligned[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low with volume confirmation
                elif close[i] < donchian_low_aligned[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # ranging or transition regime
                # In ranging market: mean revert at Donchian bands
                # Long: price touches Donchian low and reverses up with volume
                if (close[i] <= donchian_low_aligned[i] and 
                    i > start_idx and close[i-1] > donchian_low_aligned[i-1] and
                    volume_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price touches Donchian high and reverses down with volume
                elif (close[i] >= donchian_high_aligned[i] and 
                      i > start_idx and close[i-1] < donchian_high_aligned[i-1] and
                      volume_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending long when price breaks below Donchian low
                if close[i] < donchian_low_aligned[i]:
                    exit_signal = True
            else:
                # Exit ranging long when price reaches Donchian high (mean revert target)
                if close[i] >= donchian_high_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending short when price breaks above Donchian high
                if close[i] > donchian_high_aligned[i]:
                    exit_signal = True
            else:
                # Exit ranging short when price reaches Donchian low (mean revert target)
                if close[i] <= donchian_low_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals