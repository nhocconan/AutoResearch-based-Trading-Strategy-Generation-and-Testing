#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volatility filter.
- Uses 1d EMA50 for trend alignment (works in bull/bear markets)
- Donchian breakout provides clear entry/exit signals
- ATR filter ensures sufficient volatility (avoids low-vol false breakouts)
- Volume confirmation (>2.0x average) reduces whipsaw
- Position size: 0.25 (discrete level to minimize fee churn)
- Target: 20-40 trades/year for low fee drag and good generalization
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
    
    # Volume confirmation: > 2.0x 24-period average (moderate for 4h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24, 20, 14)  # EMA50, volume MA, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Volatility filter: ATR > 0.5 * 20-period ATR average (avoid extremely low vol)
        atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
        vol_filter = atr[i] > 0.5 * atr_ma[i] if not np.isnan(atr_ma[i]) else True
        
        # Donchian breakout signals
        breakout_up = close[i] > highest_high[i-1]  # Close above prior 20-period high
        breakout_down = close[i] < lowest_low[i-1]  # Close below prior 20-period low
        
        if position == 0:
            # Long: Donchian breakout up AND price > 1d EMA50 AND volume confirmation AND volatility filter
            if breakout_up and volume_confirm and vol_filter and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND price < 1d EMA50 AND volume confirmation AND volatility filter
            elif breakout_down and volume_confirm and vol_filter and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian break down OR price < 1d EMA50 (trend flip)
            if close[i] < lowest_low[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian break up OR price > 1d EMA50 (trend flip)
            if close[i] > highest_high[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_VolumeVolFilter"
timeframe = "4h"
leverage = 1.0