#!/usr/bin/env python3
name = "4h_Donchian20_Volume_Trend_1d"
timeframe = "4h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily close for trend
    close_1d = df_1d['close'].values
    # Daily EMA(20) for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Donchian channel (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume and in daily uptrend
            vol_condition = volume[i] > vol_ma[i] * 1.5
            daily_uptrend = close[i] > ema_20_1d_aligned[i]
            
            if high[i] > high_20[i] and vol_condition and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and in daily downtrend
            elif low[i] < low_20[i] and vol_condition and not daily_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low or volume drops
            if low[i] < low_20[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian high or volume drops
            if high[i] > high_20[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian(20) breakout with daily EMA(20) trend filter and volume confirmation.
# Uses Donchian channel breakouts for trend continuation and institutional interest signals.
# Daily EMA(20) ensures trades align with higher timeframe trend direction.
# Volume confirmation (1.5x average) filters false breakouts.
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Position size 0.25 balances risk and keeps trade frequency ~20-50/year.