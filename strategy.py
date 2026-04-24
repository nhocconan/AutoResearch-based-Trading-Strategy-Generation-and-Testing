#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Uses Donchian channel (20-period high/low) from prior completed 1d candles to identify breakouts.
- Breakout above upper band or below lower band with volume > 1.8x 20-bar average signals strong momentum.
- Trend filter: price must be above/below 1w EMA50 to align with higher timeframe direction.
- Designed for 1d timeframe to capture medium-term breakouts in both bull and bear markets.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 15-25 trades/year (60-100 total over 4 years) to stay fee-efficient.
- 1w EMA50 provides stable trend filter that avoids whipsaw in ranging markets.
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
    
    # Get 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need at least 21 days for Donchian(20) + 1 for shift
        return np.zeros(n)
    
    # Prior completed 1d OHLC for Donchian calculation (shifted by 1 to avoid look-ahead)
    high_1d = df_1d['high'].shift(1).values
    low_1d = df_1d['low'].shift(1).values
    
    # Donchian(20) channels: upper = 20-period high, lower = 20-period low
    # Using rolling window on prior completed 1d data
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior completed 1w close for EMA50 calculation
    close_1w = df_1w['close'].shift(1).values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to LTF (1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 50)  # Need 20 for Donchian + 30 buffer for EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above upper Donchian band AND price above 1w EMA50 AND volume confirmation
            if close[i] > donchian_upper_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower Donchian band AND price below 1w EMA50 AND volume confirmation
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below lower Donchian band OR price below 1w EMA50
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above upper Donchian band OR price above 1w EMA50
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0