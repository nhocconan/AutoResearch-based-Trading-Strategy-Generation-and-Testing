#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Long when price breaks above 6h Donchian Upper(20) AND close > 1w EMA50 (bullish trend)
- Short when price breaks below 6h Donchian Lower(20) AND close < 1w EMA50 (bearish trend)
- Volume must be > 2.0 * median volume of last 50 bars (volume confirmation to avoid fakeouts)
- Exit on opposite Donchian breakout or trend reversal (close crosses 1w EMA50)
- Uses 6h primary timeframe with 1w HTF to target 50-150 total trades over 4 years (12-37/year)
- Donchian channels provide objective breakout levels that adapt to volatility
- 1w EMA50 ensures alignment with weekly trend to avoid whipsaws in ranging markets
- Volume confirmation adapts to changing volatility, reducing noise
- Designed for BTC/ETH with edge in trending markets (breakout continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0 * median volume of last 50 bars
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian Upper, trend up (close > EMA50), volume confirmation
            if close[i] > donchian_upper[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian Lower, trend down (close < EMA50), volume confirmation
            elif close[i] < donchian_lower[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian Lower OR trend reversal (close < EMA50)
            if close[i] < donchian_lower[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian Upper OR trend reversal (close > EMA50)
            if close[i] > donchian_upper[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0