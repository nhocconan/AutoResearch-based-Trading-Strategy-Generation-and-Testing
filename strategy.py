#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 12h volume confirmation and 4h volatility regime.
Uses 4h Donchian channels (20-period) for trend direction, 12h volume spike (volume > 1.8x 20-period average)
to confirm breakout strength, and 4h volatility regime (ATR ratio < 0.7 = low volatility) to avoid false breakouts
in high volatility. Long when price breaks above 4h Donchian upper in low volatility with volume spike.
Short when price breaks below 4h Donchian lower in low volatility with volume spike.
Target: 60-120 total trades over 4 years (15-30/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume spike (volume > 1.8x 20-period average)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (vol_ma_20 * 1.8)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike.astype(float))
    
    # Get 4h data for Donchian channels and volatility regime
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 4h ATR for volatility regime
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.max([high_4h[0] - low_4h[0], np.abs(high_4h[0] - close_4h[0]), np.abs(low_4h[0] - close_4h[0])])], 
                           np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h ATR ratio (current ATR / 50-period average ATR) for volatility regime
    atr_ma_50 = pd.Series(atr_4h).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_4h / atr_ma_50
    
    # Volatility regime: ATR ratio < 0.7 = low volatility (good for breakouts)
    low_volatility = atr_ratio < 0.7
    
    atr_ratio_aligned = align_htf_to_ltf(prices, df_4h, atr_ratio)
    low_volatility_aligned = align_htf_to_ltf(prices, df_4h, low_volatility.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(low_volatility_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Donchian breakout + volume spike + low volatility
        breakout_long = close[i] > donchian_high_aligned[i]
        breakout_short = close[i] < donchian_low_aligned[i]
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        vol_regime = low_volatility_aligned[i] > 0.5  # True if low volatility
        
        long_entry = breakout_long and vol_confirm and vol_regime
        short_entry = breakout_short and vol_confirm and vol_regime
        
        # Exit when price returns to opposite Donchian level (mean reversion within channel)
        exit_long = position == 1 and close[i] < donchian_low_aligned[i]
        exit_short = position == -1 and close[i] > donchian_high_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_vol_volatility_v1"
timeframe = "4h"
leverage = 1.0