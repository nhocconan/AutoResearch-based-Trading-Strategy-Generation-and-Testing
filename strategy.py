#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian breakout with 1w volume confirmation and weekly volatility regime.
Uses 1-week Donchian channels for trend direction, 1w volume spike (volume > 1.5x 20-period average) 
to confirm breakout strength, and 1-week volatility regime (ATR ratio < 0.8 = low volatility) 
to avoid false breakouts in high volatility. Long when price breaks above weekly Donchian upper 
in low volatility with volume spike. Short when price breaks below weekly Donchian lower in 
low volatility with volume spike. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for price action (close prices for breakout)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Get 1w data for Donchian channels and volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1-week Donchian channels (20-period)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate 1w volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1w > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1w, vol_spike.astype(float))
    
    # Calculate 1-week ATR for volatility regime
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], 
                           np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1-week ATR ratio (current ATR / 50-period average ATR) for volatility regime
    atr_ma_50 = pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1w / atr_ma_50
    
    # Volatility regime: ATR ratio < 0.8 = low volatility (good for breakouts)
    low_volatility = atr_ratio < 0.8
    
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    low_volatility_aligned = align_htf_to_ltf(prices, df_1w, low_volatility.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
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

name = "1d_1w_donchian_vol_volatility"
timeframe = "1d"
leverage = 1.0