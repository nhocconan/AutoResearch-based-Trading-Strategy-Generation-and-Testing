#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR filter and volume confirmation.
- Uses 6h timeframe (primary) and 1d HTF for ATR-based volatility filter
- Donchian channels calculated from prior 20 periods to avoid look-ahead
- Breakout logic: long when price crosses above upper band with volume confirmation and ATR > 1.2 * ATR_MA(50), short when price crosses below lower band with same conditions
- ATR filter ensures we only trade during sufficient volatility regimes, avoiding choppy markets
- Volume confirmation: current volume > 1.5 * 20-period volume MA to avoid low-volume false signals
- Exit: reverse signal or when price reverts to the 20-period moving average (mean reversion to midpoint)
- Discrete signal size: 0.25 to balance return and risk
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
- Works in both bull/bear: volatility filter avoids low-volatility whipsaws, Donchian breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels (prior periods only)
    # Use rolling window with shift(1) to ensure we only use completed periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donch_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    donch_mid = (donch_high + donch_low) / 2  # Midpoint for exit
    
    # Calculate 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # ATR(14) - using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR and ATR_MA to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Volatility filter: ATR > 1.2 * ATR_MA(50) ensures sufficient volatility
    vol_filter = atr_14_aligned > (1.2 * atr_ma_50_aligned)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need Donchian(20) and ATR_MA(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or
            np.isnan(volume_confirm[i]) or np.isnan(vol_filter[i]) or
            np.isnan(donch_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above upper Donchian band AND volatility filter AND volume confirmation
            if close[i] > donch_high[i] and close[i-1] <= donch_high[i-1] and vol_filter[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below lower Donchian band AND volatility filter AND volume confirmation
            elif close[i] < donch_low[i] and close[i-1] >= donch_low[i-1] and vol_filter[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to Donchian midpoint (mean reversion) or reverse signal
            if close[i] <= donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to Donchian midpoint (mean reversion) or reverse signal
            if close[i] >= donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dATR_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0