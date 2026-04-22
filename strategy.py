#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian breakout captures strong momentum moves in both bull and bear markets.
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades.
# Volume > 1.5x average confirms breakout strength.
# Works in both bull and bear markets by following the weekly trend direction.
# Uses discrete position sizing (0.25) to minimize fee churn.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data for EMA trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1d Donchian(20) - upper and lower bands
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average on 1d
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(df_1d['close'], df_1w, ema_50_1w)
    
    # Align Donchian bands to 1d timeframe (they're already on 1d, but ensure proper indexing)
    # Since we're working with 1d data directly, we can use the arrays as-is
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index 20 to ensure Donchian bands are calculated
    for i in range(20, len(df_1d)):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                # Map 1d signal back to original 15m index (if needed)
                # Since we're using 1d as primary timeframe, we need to map to original
                pass
            continue
        
        # Map 1d index to original index (assuming original is 15m)
        # For 1d timeframe, each 1d bar = 96 * 15m bars
        orig_idx = i * 96
        
        if orig_idx >= n:
            break
            
        if position == 0:
            # Long: Price breaks above Donchian high + above weekly EMA + volume spike
            if close_1d[i] > donch_high[i] and close_1d[i] > ema_50_1w_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[orig_idx] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below weekly EMA + volume spike
            elif close_1d[i] < donch_low[i] and close_1d[i] < ema_50_1w_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[orig_idx] = -0.25
                position = -1
        else:
            # Exit: Price crosses back through Donchian levels or weekly EMA
            if position == 1:
                # Exit long: Price below Donchian low or below weekly EMA
                if close_1d[i] < donch_low[i] or close_1d[i] < ema_50_1w_aligned[i]:
                    signals[orig_idx] = 0.0
                    position = 0
                else:
                    signals[orig_idx] = 0.25
            else:  # position == -1
                # Exit short: Price above Donchian high or above weekly EMA
                if close_1d[i] > donch_high[i] or close_1d[i] > ema_50_1w_aligned[i]:
                    signals[orig_idx] = 0.0
                    position = 0
                else:
                    signals[orig_idx] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0