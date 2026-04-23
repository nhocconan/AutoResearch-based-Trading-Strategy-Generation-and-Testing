#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
- Donchian channels act as adaptive support/resistance on 4h chart
- Breakout above upper band with volume > 1.8x average signals bullish momentum
- Breakdown below lower band with volume > 1.8x average signals bearish momentum
- 1d EMA50 ensures trades align with higher timeframe trend (avoid counter-trend)
- Discrete position size 0.25 to minimize drawdown during crashes like 2022
- Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years)
- Works in both bull/bear via 1d trend filter and volatility-adjusted breakouts
- Uses volume confirmation to avoid false breakouts and reduce overtrading
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
    
    # Donchian channel (20-period) using prior bar's data to avoid look-ahead
    high_shifted = np.roll(high, 1)
    low_shifted = np.roll(low, 1)
    high_shifted[0] = np.nan
    low_shifted[0] = np.nan
    
    upper_band = pd.Series(high_shifted).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_shifted).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Close > upper band AND price above 1d EMA50 AND volume confirmation
            if close[i] > upper_band[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close < lower band AND price below 1d EMA50 AND volume confirmation
            elif close[i] < lower_band[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < lower band OR price crosses below 1d EMA50
            if close[i] < lower_band[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > upper band OR price crosses above 1d EMA50
            if close[i] > upper_band[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0