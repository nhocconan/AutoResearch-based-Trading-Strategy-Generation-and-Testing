#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation.
- Long when price breaks above Donchian upper (20-period high) AND ATR(14) > 0.8 * median ATR(14) of last 50 bars (volatility filter to avoid low-vol fakeouts)
- Short when price breaks below Donchian lower (20-period low) AND ATR(14) > 0.8 * median ATR(14) of last 50 bars
- Volume must be > 1.5 * median volume of last 20 bars (volume confirmation to avoid fakeouts)
- Exit on opposite Donchian breakout or when ATR(14) < 0.5 * median ATR(14) of last 50 bars (volatility collapse exit)
- Uses 4h primary timeframe with 1d HTF for ATR calculation to ensure higher timeframe volatility context
- Donchian channels provide clear structural breakouts with defined risk
- ATR filter ensures we only trade during sufficient volatility regimes, reducing whipsaws in low-vol markets
- Volume confirmation adds confluence to avoid low-liquidity breakouts
- Designed for BTC/ETH with edge in both trending (breakout continuation) and volatile ranging (mean reversion at extremes) markets
- Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
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
    
    # Calculate Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ATR(14) to 4h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volatility filter: ATR > 0.8 * median ATR of last 50 bars
    atr_median = pd.Series(atr_14_1d_aligned).rolling(window=50, min_periods=50).median().values
    vol_filter = atr_14_1d_aligned > (0.8 * atr_median)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    # Volatility collapse exit: ATR < 0.5 * median ATR of last 50 bars
    vol_exit = atr_14_1d_aligned < (0.5 * atr_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_median[i]) or 
            np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper, volatility filter, volume confirmation
            if close[i] > donchian_upper[i] and vol_filter[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, volatility filter, volume confirmation
            elif close[i] < donchian_lower[i] and vol_filter[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower OR volatility collapse
            if close[i] < donchian_lower[i] or vol_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper OR volatility collapse
            if close[i] > donchian_upper[i] or vol_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0