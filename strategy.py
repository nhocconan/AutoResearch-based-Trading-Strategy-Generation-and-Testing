#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
Long when price breaks above Donchian upper band AND 1d ATR(14) < 30-period median ATR (low volatility regime) AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band AND 1d ATR(14) < 30-period median ATR AND volume > 1.5x 20-period average.
Exit when price touches the opposite Donchian band.
Uses 1d HTF for volatility regime filter to avoid whipsaws in high volatility markets. Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Calculate 1d ATR(14) for volatility filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) - Wilder's smoothing
    atr_14 = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.nanmean(tr[1:i+1])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # 30-period median ATR for volatility regime
    atr_median_30 = pd.Series(atr_14).rolling(window=30, min_periods=30).median().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_median_30_aligned = align_htf_to_ltf(prices, df_1d, atr_median_30)
    
    # Low volatility regime: current ATR < median ATR
    low_vol_regime = atr_14_aligned < atr_median_30_aligned
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 30, 20)  # Donchian (20), ATR median (30), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(low_vol_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        up = upper[i]
        lo = lower[i]
        vol_ma_val = vol_ma[i]
        is_low_vol = low_vol_regime[i]
        
        if position == 0:
            # Long: Break above Donchian upper AND low volatility regime AND volume spike
            if price > up and is_low_vol and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND low volatility regime AND volume spike
            elif price < lo and is_low_vol and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower band
                if price < lo:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper band
                if price > up:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dATR_VolRegime_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0