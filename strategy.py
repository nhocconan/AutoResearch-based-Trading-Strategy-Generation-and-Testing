#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
- Uses Donchian channel (20-period high/low) from prior completed 1d candles for breakout detection.
- Breakout above upper band or below lower band with volume > 1.5x 20-bar average signals strong momentum.
- Trend filter: price must be above/below 1w EMA34 to align with higher timeframe direction.
- Designed for 1d timeframe to capture medium-term breakouts in both bull and bear markets.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 15-25 trades/year (60-100 total over 4 years) to stay fee-efficient.
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
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior completed 1w close for EMA34
    close_1w = df_1w['close'].shift(1).values
    
    # 1w EMA34 trend filter
    close_1w_series = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w_series).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to LTF
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Donchian(20) from prior completed 1d candles
    high_shift = pd.Series(high).shift(1).values
    low_shift = pd.Series(low).shift(1).values
    donchian_high = pd.Series(high_shift).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_shift).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above Donchian high AND price above 1w EMA34 AND volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND price below 1w EMA34 AND volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below Donchian low OR price below 1w EMA34
            if close[i] < donchian_low[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above Donchian high OR price above 1w EMA34
            if close[i] > donchian_high[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0