#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
Long when price breaks above Donchian upper band AND 1d ATR(14) > 1.2x its 50-period MA (high volatility regime) AND volume > 1.5x 20-period MA.
Short when price breaks below Donchian lower band AND 1d ATR(14) > 1.2x its 50-period MA AND volume > 1.5x 20-period MA.
Exit when price touches opposite Donchian band.
Uses 1d HTF for volatility filter to ensure breakouts occur in high momentum environments, reducing false signals.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Volatility filter avoids low-momentum choppy markets where breakouts fail, improving win rate in both bull and bear regimes.
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
    
    # Calculate 12h Donchian channels (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(20, n):
        # Use lookback of 20 periods (excluding current bar to avoid look-ahead)
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
    
    # Calculate 1d ATR(14) for volatility filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need 50 for ATR MA
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with 1d indices
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR(14) 50-period MA
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    # Volatility filter: ATR(14) > 1.2x its 50-period MA
    vol_filter_1d = atr_14 > (1.2 * atr_ma_50)
    vol_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_filter_1d.astype(float))
    
    # Calculate 12h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Donchian (needs 20), ATR MA (needs 50), volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_filter_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        vol_filter_val = vol_filter_1d_aligned[i] > 0.5  # Convert to boolean
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 12h volume > 1.5x 20-period MA
        vol_spike = volume[i] > (1.5 * vol_ma_val)
        
        if position == 0:
            # Long: Break above Donchian upper AND volatility filter AND volume spike
            if price > upper and vol_filter_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND volatility filter AND volume spike
            elif price < lower and vol_filter_val and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Donchian lower (opposite band)
                if price < lower:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Donchian upper (opposite band)
                if price > upper:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dATR_VolFilter_VolumeSpike"
timeframe = "12h"
leverage = 1.0