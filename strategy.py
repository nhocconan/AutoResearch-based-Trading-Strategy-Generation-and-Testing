#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume spike confirmation.
Long when price breaks above Donchian upper(20) AND 1d ATR(14) > 1.5x 50-period MA AND volume > 2.0x 20-period MA.
Short when price breaks below Donchian lower(20) AND 1d ATR(14) > 1.5x 50-period MA AND volume > 2.0x 20-period MA.
Exit when price touches opposite Donchian level or 1d ATR(14) falls below 1.0x 50-period MA.
Uses 1d HTF for volatility regime filter to avoid low-volatility false breakouts, volume spike for momentum confirmation.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Donchian channels provide structure, 1d ATR filter ensures breakouts occur in sufficient volatility,
volume spike avoids low-momentum breakouts. Works in bull (breakouts with volume) and bear (volatile breakdowns).
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
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_upper[i] = np.max(high[i-lookback+1:i+1])
        donchian_lower[i] = np.min(low[i-lookback+1:i+1])
    
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
    
    # ATR(14) - Welles Wilder's smoothing
    atr_14 = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.nanmean(tr[1:15])  # First ATR is average of first 14 TR
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # 50-period MA of ATR(14)
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Calculate 1h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 50 + 14, 20)  # Donchian, ATR(14)+MA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        atr_val = atr_14_aligned[i]
        atr_ma_val = atr_ma_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volatility filter: ATR(14) > 1.5x 50-period MA (ensures sufficient volatility)
        vol_regime = atr_val > 1.5 * atr_ma_val
        
        # Volume filter: 1h volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Break above Donchian upper AND volatility regime AND volume filter
            if price > upper and vol_regime and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND volatility regime AND volume filter
            elif price < lower and vol_regime and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower Donchian OR volatility regime ends (ATR < 1.0x MA)
                if price < lower or atr_val < 1.0 * atr_ma_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper Donchian OR volatility regime ends (ATR < 1.0x MA)
                if price > upper or atr_val < 1.0 * atr_ma_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dATR_VolRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0