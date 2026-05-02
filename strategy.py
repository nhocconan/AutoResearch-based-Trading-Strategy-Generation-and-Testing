#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ADX regime filter + volume confirmation
# Donchian breakout provides clear entry/exit with structure
# 1d ADX > 25 = trending (follow breakout direction), ADX < 20 = ranging (fade breakout)
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by adapting to regime via ADX
# Uses 1d for HTF regime and ATR calculation for stability

name = "4h_Donchian20_1dADXRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX regime and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
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
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    high_roll = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max()
    low_roll = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min()
    donchian_high = high_roll.shift(1).values  # breakout of previous period
    donchian_low = low_roll.shift(1).values
    
    # Calculate 4x volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, ADX and volume MA)
    start_idx = max(donchian_window, 20) + 5  # 25
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from 1d ADX
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # In trending market: follow Donchian breakout direction
                # Long: price breaks above Donchian high
                if close[i] > donchian_high[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low
                elif close[i] < donchian_low[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # ranging or transition regime
                # In ranging market: fade Donchian breakouts (mean reversion)
                # Long: price breaks below Donchian low then reverses (oversold bounce)
                # Short: price breaks above Donchian high then reverses (overbought fade)
                if (close[i] < donchian_low[i] and 
                    i > start_idx and close[i-1] >= donchian_low[i-1] and
                    volume_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                elif (close[i] > donchian_high[i] and 
                      i > start_idx and close[i-1] <= donchian_high[i-1] and
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
                if close[i] < donchian_low[i]:
                    exit_signal = True
            else:
                # Exit ranging long when price reaches midpoint (mean reversion target)
                midpoint = (donchian_high[i] + donchian_low[i]) / 2
                if close[i] >= midpoint:
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
                if close[i] > donchian_high[i]:
                    exit_signal = True
            else:
                # Exit ranging short when price reaches midpoint (mean reversion target)
                midpoint = (donchian_high[i] + donchian_low[i]) / 2
                if close[i] <= midpoint:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals