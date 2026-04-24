#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1-week ATR regime filter.
- Primary: 1d for execution, HTF: 1w for volatility regime classification.
- ATR(14) on 1w: high ATR = volatile/trending (favor breakouts), low ATR = ranging (favor mean reversion).
- Entry: In volatile regime (ATR > 1.5 * 20w ATR MA): breakout long/short on Donchian touch.
         In ranging regime (ATR <= 1.5 * 20w ATR MA): mean reversion at Donchian extremes with reversal.
- Exit: Opposite Donchian breakout or regime shift.
- Volume confirmation: current volume > 1.2 * 20d volume MA to filter weak breakouts.
- Discrete signal size: 0.25 to balance opportunity and drawdown.
- Target: 30-100 trades over 4 years (7-25/year) for 1d timeframe.
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
    
    # Get 1w data for ATR-based regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # ATR(14) on 1w
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['low'].shift())).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 20-period ATR MA for regime threshold
    atr_ma_20w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    volatile_regime = atr_1w > (1.5 * atr_ma_20w)  # True = volatile/trending
    
    # Align regime and ATR to 1d
    volatile_aligned = align_htf_to_ltf(prices, df_1w, volatile_regime.astype(float))
    
    # Donchian channels (20-period) on 1d
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: current volume > 1.2 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.2 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, lookback, 20)  # Need enough 1w bars for ATR/MA and lookback for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(volatile_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_volatile = volatile_aligned[i] > 0.5
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            if volume_spike[i]:
                if is_volatile:  # Volatile regime: breakout strategy
                    if curr_close > highest_high[i]:
                        signals[i] = 0.25
                        position = 1
                    elif curr_close < lowest_low[i]:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging regime: mean reversion at extremes
                    if curr_low <= lowest_low[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    elif curr_high >= highest_high[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            if curr_close < donchian_mid[i] or not is_volatile:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if curr_close > donchian_mid[i] or not is_volatile:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wATRRegime_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0