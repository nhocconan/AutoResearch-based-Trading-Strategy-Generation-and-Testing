#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
- Long when price breaks above 20-day high AND close > 1w EMA34 (bullish trend)
- Short when price breaks below 20-day low AND close < 1w EMA34 (bearish trend)
- Volume must be > 1.5 * median volume of last 20 days (volume confirmation)
- Exit on opposite Donchian breakout or trend reversal (close crosses 1w EMA34)
- Uses 1d primary timeframe with 1w HTF to target 30-100 total trades over 4 years (7-25/year)
- Donchian channels provide robust trend-following structure that works in both bull and bear markets
- 1w EMA34 ensures alignment with higher timeframe trend to avoid whipsaws during choppy periods
- Volume confirmation filters out low-conviction breakouts
- Designed for BTC/ETH with edge in capturing sustained moves while avoiding false breakouts
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
    
    # Calculate Donchian channels (20-day high/low)
    # Using rolling window on daily data directly
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 days
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high, trend up (close > 1w EMA34), volume confirmation
            if close[i] > high_20[i] and close[i] > ema_34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low, trend down (close < 1w EMA34), volume confirmation
            elif close[i] < low_20[i] and close[i] < ema_34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 20-day low OR trend reversal (close < 1w EMA34)
            if close[i] < low_20[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 20-day high OR trend reversal (close > 1w EMA34)
            if close[i] > high_20[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0