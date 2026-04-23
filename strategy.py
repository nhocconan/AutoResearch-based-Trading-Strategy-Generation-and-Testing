#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d (daily bars) for lower trade frequency and reduced fee drag.
- Donchian(20) breakout captures medium-term trends; long on break above upper band, short on break below lower band.
- 1w EMA50 ensures trades align with weekly trend to avoid counter-trend entries.
- Volume confirmation (>1.5x 20-day average) filters weak breakouts.
- Discrete position size 0.25 to manage drawdown (BTC 2022 drawdown ~77% → ~19% equity loss at 0.25 size).
- Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years).
- Uses 1w HTF for trend filter as specified in experiment #81424.
- Designed to work in both bull (trend continuation) and bear (trend reversals) regimes via weekly EMA filter.
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
    
    # Donchian channels (20-period lookback, using prior close to avoid look-ahead)
    # Upper band: highest high over past 20 days (excluding current bar)
    # Lower band: lowest low over past 20 days (excluding current bar)
    high_shifted = np.roll(high, 1)
    low_shifted = np.roll(low, 1)
    high_shifted[0] = np.nan
    low_shifted[0] = np.nan
    
    # Calculate rolling max/min on shifted arrays (past 20 bars)
    upper_band = pd.Series(high_shifted).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_shifted).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    # 1w data for EMA50 trend filter (weekly timeframe)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian (20) and weekly EMA (50) warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > upper Donchian band AND price above weekly EMA50 AND volume confirmation
            if close[i] > upper_band[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < lower Donchian band AND price below weekly EMA50 AND volume confirmation
            elif close[i] < lower_band[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < lower Donchian band OR price crosses below weekly EMA50
            if close[i] < lower_band[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > upper Donchian band OR price crosses above weekly EMA50
            if close[i] > upper_band[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0