#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR regime filter.
- Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND volume > 2.0x 20-period average AND ATR > 0.7x 50-period ATR average.
- Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND volume > 2.0x 20-period average AND ATR > 0.7x 50-period ATR average.
- Exit when price crosses the Donchian midpoint (10-period average of high/low) OR price crosses 1d EMA50 in opposite direction.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Designed for ~20-30 trades/year (80-120 total over 4 years) to stay within fee-efficient range.
- Combines proven elements: Donchian structure + trend filter + volume/volatility confirmation.
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d OHLC (completed daily bar)
    close_1d = df_1d['close'].shift(1).values
    
    # Align to 4h timeframe
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility regime filter
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # ATR ratio: current ATR / 50-period average (avoid low volatility chop)
    atr_ma_long = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / np.where(atr_ma_long > 0, atr_ma_long, 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 50, atr_period, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average) + ATR ratio > 0.7 (avoid low vol)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        vol_regime = atr_ratio[i] > 0.7
        
        if position == 0:
            # Long: Close > Donchian high AND price above 1d EMA50 AND volume confirmation AND vol regime
            if close[i] > highest_high[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: Close < Donchian low AND price below 1d EMA50 AND volume confirmation AND vol regime
            elif close[i] < lowest_low[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm and vol_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < Donchian mid OR price crosses below 1d EMA50
            if close[i] < donchian_mid[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > Donchian mid OR price crosses above 1d EMA50
            if close[i] > donchian_mid[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_VolumeATR_Filter_v1"
timeframe = "4h"
leverage = 1.0