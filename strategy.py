#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout (20) for trend direction,
# 1d ADX regime filter (ADX>25 trending, ADX<20 ranging), and 1h volume confirmation (1.5x 20-bar avg).
# In trending regime (ADX>25): trade Donchian breakouts in direction of trend.
# In ranging regime (ADX<20): fade Donchian breakouts (mean reversion at extremes).
# Uses discrete position sizing 0.20 to limit risk and minimize fee churn.
# Targets 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits.
# Session filter (08-20 UTC) reduces noise trades outside active market hours.
# Works in both bull and bear markets by adapting to regime via ADX.

name = "1h_Donchian20_4hTrend_1dADXRegime_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Donchian channels (trend structure)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    highest_4h = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    lowest_4h = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian channels to 1h
    highest_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_4h)
    lowest_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_4h)
    
    # Load 1d data ONCE before loop for ADX regime filter
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
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, ADX and volume MA)
    start_idx = 50  # max(20 for Donchian, 34 for ADX, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Skip if outside active session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(highest_4h_aligned[i]) or np.isnan(lowest_4h_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from 1d ADX
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # In trending market: trade Donchian breakouts in direction of trend
                # Long: price breaks above 4h Donchian high AND previous close <= that high
                if (close[i] > highest_4h_aligned[i] and 
                    i > start_idx and close[i-1] <= highest_4h_aligned[i-1] and
                    volume_confirm[i]):
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks below 4h Donchian low AND previous close >= that low
                elif (close[i] < lowest_4h_aligned[i] and 
                      i > start_idx and close[i-1] >= lowest_4h_aligned[i-1] and
                      volume_confirm[i]):
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # ranging or transition regime
                # In ranging market: fade Donchian breakouts (mean reversion at extremes)
                # Long: price breaks below 4h Donchian low AND previous close >= that low (oversold bounce)
                if (close[i] < lowest_4h_aligned[i] and 
                    i > start_idx and close[i-1] >= lowest_4h_aligned[i-1] and
                    volume_confirm[i]):
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks above 4h Donchian high AND previous close <= that high (overbought fade)
                elif (close[i] > highest_4h_aligned[i] and 
                      i > start_idx and close[i-1] <= highest_4h_aligned[i-1] and
                      volume_confirm[i]):
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending long when price returns to or below 4h Donchian middle
                donchian_middle = (highest_4h_aligned[i] + lowest_4h_aligned[i]) / 2.0
                if close[i] <= donchian_middle:
                    exit_signal = True
            else:
                # Exit ranging long when price moves back above 4h Donchian low (reversion complete)
                if close[i] > lowest_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending short when price returns to or above 4h Donchian middle
                donchian_middle = (highest_4h_aligned[i] + lowest_4h_aligned[i]) / 2.0
                if close[i] >= donchian_middle:
                    exit_signal = True
            else:
                # Exit ranging short when price moves back below 4h Donchian high (reversion complete)
                if close[i] < highest_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals