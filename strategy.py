#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation.
Long when price breaks above Donchian upper band AND 1d ATR(14) > 1.5x 50-period MA AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band AND 1d ATR(14) > 1.5x 50-period MA AND volume > 1.5x 20-period average.
Exit when price touches the opposite Donchian band.
Uses 1d HTF for ATR-based volatility regime filter to avoid ranging markets. Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Calculate 1d ATR(14) for volatility regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14_1d = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14_1d[i] = np.nanmean(tr[1:15])  # First ATR is simple average
        else:
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # 50-period MA of ATR for volatility regime threshold
    atr_ma_50 = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
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
    start_idx = max(lookback - 1, 64, 20)  # Donchian (20), ATR(14)+MA(50) needs ~64 bars
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_ma_50_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_val = atr_14_aligned[i]
        atr_ma_val = atr_ma_50_aligned[i]
        up = upper[i]
        lo = lower[i]
        vol_ma_val = vol_ma[i]
        
        # Volatility regime: ATR > 1.5x its 50-period MA (avoid ranging markets)
        vol_regime = atr_val > 1.5 * atr_ma_val
        
        if position == 0:
            # Long: Break above Donchian upper AND volatility regime AND volume spike
            if price > up and vol_regime and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND volatility regime AND volume spike
            elif price < lo and vol_regime and volume[i] > 1.5 * vol_ma_val:
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