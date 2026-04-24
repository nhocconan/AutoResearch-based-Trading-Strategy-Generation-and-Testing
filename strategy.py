#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
- Uses Donchian channel (20-period high/low) from prior completed 1d candles.
- Breakout above upper band or below lower band with volume > 2.0x 20-bar average signals strong momentum.
- Trend filter: price must be above/below 1w EMA34 to align with higher timeframe direction.
- Designed for 1d timeframe to capture medium-term breakouts in both bull and bear markets.
- Uses discrete position size 0.30 to limit drawdown and reduce fee churn.
- Targets 15-30 trades/year (60-120 total over 4 years) to stay fee-efficient.
- Based on proven pattern: Donchian breakout + volume + trend filter showed strong performance in DB.
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period) from prior completed 1d candles
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Upper band: 20-period high, Lower band: 20-period low
    upper_band = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 1d timeframe (wait for 1d bar to close)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above upper band AND price above 1w EMA34 AND volume confirmation
            if close[i] > upper_band_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.30
                position = 1
            # Short: breakout below lower band AND price below 1w EMA34 AND volume confirmation
            elif close[i] < lower_band_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: close below lower band OR price below 1w EMA34
            if close[i] < lower_band_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: close above upper band OR price above 1w EMA34
            if close[i] > upper_band_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0