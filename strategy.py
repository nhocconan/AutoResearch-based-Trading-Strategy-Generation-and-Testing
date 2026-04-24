#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Uses 12h timeframe (primary) and 1d HTF for EMA34 trend alignment
- Donchian levels calculated from prior 20-period 12h high/low: upper = max(high[-20:]), lower = min(low[-20:])
- Breakout logic: long when price crosses above upper band with volume confirmation and uptrend (price > 1d EMA34), short when price crosses below lower band with volume confirmation and downtrend (price < 1d EMA34)
- Trend filter: only long when price > 1d EMA34, only short when price < 1d EMA34
- Volume confirmation: current volume > 1.5 * 20-period volume MA to avoid low-volume false signals
- Discrete signal size: 0.25 to balance capture and fee drag
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture momentum in all regimes
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate prior 20-period 12h Donchian channels
    # Need at least 20 bars of 12h data for initialization
    high_12h = high
    low_12h = low
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Trend filter: price above/below 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 20)  # Donchian(20), EMA34, volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above upper band AND uptrend AND volume confirmation
            if close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below lower band AND downtrend AND volume confirmation
            elif close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below lower band (mean reversion) or reverse signal
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above upper band (mean reversion) or reverse signal
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0