#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation.
Long when price breaks above Donchian upper band AND 1d ATR(14) > 1.5x 50-period SMA of ATR AND volume > 1.2x 20-period average.
Short when price breaks below Donchian lower band AND same ATR/volume conditions.
Exit when price touches the opposite Donchian band or ATR(14) < 1.2x 50-period SMA of ATR.
Uses 1d HTF for ATR regime filter to avoid false breakouts in low-volatility environments.
Target: 75-200 total trades over 4 years (19-50/year).
Donchian channels provide clear breakout levels; ATR filter ensures we only trade during sufficient volatility regimes.
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
    
    # Calculate 1d ATR(14) for volatility regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = high_1d[0] - close_1d[0]  # First period
    tr3[0] = low_1d[0] - close_1d[0]   # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - Wilder's smoothing
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[0:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # 50-period SMA of ATR(14)
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR and its MA to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Calculate 4h Donchian(20) channels
    donchian_window = 20
    upper_band = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_band = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 50, 20)  # Donchian(20), ATR MA(50), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = upper_band[i]
        lower = lower_band[i]
        atr_val = atr_14_aligned[i]
        atr_ma_val = atr_ma_50_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Volatility regime filter: ATR(14) > 1.5x 50-period SMA of ATR
        vol_regime = atr_val > 1.5 * atr_ma_val
        # Low vol exit: ATR(14) < 1.2x 50-period SMA of ATR
        low_vol_exit = atr_val < 1.2 * atr_ma_val
        # Volume confirmation
        vol_spike = volume[i] > 1.2 * vol_ma_val
        
        if position == 0:
            # Long: Break above upper band AND volatility regime AND volume spike
            if price > upper and vol_regime and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band AND volatility regime AND volume spike
            elif price < lower and vol_regime and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower band OR low volatility regime
                if price < lower or low_vol_exit:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper band OR low volatility regime
                if price > upper or low_vol_exit:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dATRregime_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0