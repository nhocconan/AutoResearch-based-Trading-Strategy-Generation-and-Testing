#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + 1w ADX regime filter
# Donchian breakout captures strong momentum moves in both bull and bear markets
# 1d volume spike (2.0x 20-period average) ensures institutional participation
# 1w ADX > 25 filters for trending regimes where breakouts work best
# Uses discrete position sizing 0.25 to minimize fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by only taking breakouts in trending regimes (ADX > 25)

name = "12h_Donchian20_1dVolumeSpike_1wADXTrend_v1"
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
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 1d volume spike (2.0x 20-period average)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (vol_ma_1d * 2.0)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 1w ADX for regime filter (trending vs ranging)
    # True Range
    tr1 = np.abs(df_1w['high'].values - df_1w['low'].values)
    tr2 = np.abs(df_1w['high'].values - np.roll(df_1w['close'].values, 1))
    tr3 = np.abs(df_1w['low'].values - np.roll(df_1w['close'].values, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((df_1w['high'].values - np.roll(df_1w['high'].values, 1)) > 
                       (np.roll(df_1w['low'].values, 1) - df_1w['low'].values),
                       np.maximum(df_1w['high'].values - np.roll(df_1w['high'].values, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1w['low'].values, 1) - df_1w['low'].values) > 
                        (df_1w['high'].values - np.roll(df_1w['high'].values, 1)),
                        np.maximum(np.roll(df_1w['low'].values, 1) - df_1w['low'].values, 0), 0)
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
    
    # Align 1w ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 12h Donchian channels (20-period)
    # Need at least 20 periods for Donchian, plus warmup for other indicators
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, volume spike, and ADX)
    start_idx = 60  # max(20 for Donchian, 20 for volume MA, 34 for ADX) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from 1w ADX
        trending = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # In trending market: take Donchian breakouts with volume confirmation
                # Long: price breaks above Donchian upper channel
                if (close[i] > highest_high[i] and 
                    volume_spike_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian lower channel
                elif (close[i] < lowest_low[i] and 
                      volume_spike_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # In ranging regime: no entries (avoid false breakouts)
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when price returns to the middle of the Donchian channel
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] < donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price returns to the middle of the Donchian channel
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] > donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals