#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_v1
# Strategy: 4h Donchian breakout with 1d volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture momentum in trending markets. Volume confirmation ensures breakout validity. Works in both bull and bear by trading breakouts in the direction of the 1d trend (using EMA50). Designed for low trade frequency to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian channels on 4h (20-period)
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    upper_band = highest_high.values
    lower_band = lowest_low.values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 20-period average
        # Note: current 1d volume is not directly available, so we use the aligned value
        # which represents the last completed 1d bar's volume average
        vol_confirm = volume_1d[-1] > vol_avg_20_1d_aligned[i] if len(volume_1d) > 0 else False
        # Simplified: use the aligned volume average as threshold for current bar
        # More practical: check if current volume is above average
        # We approximate by comparing current 4h volume to a threshold, but better to use 1d
        # Since we don't have current 1d volume, we skip vol_confirm and rely on breakout + trend
        # Instead, we use the fact that vol_avg_20_1d_aligned is based on completed bars
        # and assume current volume is reflective if we see a breakout
        # For simplicity, we'll use a placeholder: always allow if breakout occurs
        # In practice, we could use the current 4h volume vs its own average, but per instructions
        # we should use 1d volume. We'll use the last available 1d volume (aligned) vs its average
        # Get the last completed 1d bar's volume (aligned value represents the average, not the volume)
        # This is a limitation; we approximate by using the close price's alignment
        # Better approach: use the 1d volume series directly
        vol_1d_series = df_1d['volume'].values
        if len(vol_1d_series) == 0:
            vol_confirm = False
        else:
            # Align the raw 1d volume to 4h
            vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_series)
            vol_confirm = vol_1d_aligned[i] > vol_avg_20_1d_aligned[i] if not np.isnan(vol_1d_aligned[i]) else False
        
        # Trend filter: close vs 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Price breaks above upper band AND uptrend AND volume confirmation
        if not np.isnan(upper_band[i]) and close[i] > upper_band[i] and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price breaks below lower band AND downtrend AND volume confirmation
        elif not np.isnan(lower_band[i]) and close[i] < lower_band[i] and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price crosses opposite band
        elif position == 1 and close[i] < lower_band[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > upper_band[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals