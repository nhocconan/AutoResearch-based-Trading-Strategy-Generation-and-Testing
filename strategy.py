#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and 1d volatility regime.
Long when price breaks above H3 pivot level with volume spike and ATR ratio < 0.8 (low volatility).
Short when price breaks below L3 pivot level with volume spike and ATR ratio < 0.8.
Uses 1d Camarilla levels from previous day, volume > 1.5x 20-period average, and ATR ratio < 0.8.
Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag and avoid overtrading.
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
    
    # Get 1d data for Camarilla pivots, volume, and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day Camarilla pivot levels (based on previous day)
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), H2 = close + 0.5*(high-low)
    # L2 = close - 0.5*(high-low), L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    high_low = high_1d - low_1d
    camarilla_h3 = close_1d + 1.0 * high_low
    camarilla_l3 = close_1d - 1.0 * high_low
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate 1d ATR for volatility regime
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                           np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR ratio (current ATR / 50-period average ATR) for volatility regime
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / atr_ma_50
    
    # Volatility regime: ATR ratio < 0.8 = low volatility (good for breakouts)
    low_volatility = atr_ratio < 0.8
    
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    low_volatility_aligned = align_htf_to_ltf(prices, df_1d, low_volatility.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(low_volatility_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Camarilla breakout + volume spike + low volatility
        breakout_long = close[i] > camarilla_h3_aligned[i]
        breakout_short = close[i] < camarilla_l3_aligned[i]
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        vol_regime = low_volatility_aligned[i] > 0.5  # True if low volatility
        
        long_entry = breakout_long and vol_confirm and vol_regime
        short_entry = breakout_short and vol_confirm and vol_regime
        
        # Exit when price returns to opposite Camarilla level (mean reversion within range)
        exit_long = position == 1 and close[i] < camarilla_l3_aligned[i]
        exit_short = position == -1 and close[i] > camarilla_h3_aligned[i]
        
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

name = "4h_1d_camarilla_breakout_vol_regime"
timeframe = "4h"
leverage = 1.0