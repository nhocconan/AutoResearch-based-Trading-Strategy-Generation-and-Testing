#!/usr/bin/env python3
name = "12h_Keltner_Breakout_Volume_1wTrend"
timeframe = "12h"
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
    
    # ATR(20) for Keltner channels
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_series = pd.Series(tr)
    atr = tr_series.rolling(window=20, min_periods=20).mean().values
    
    # EMA(20) for Keltner center
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner upper and lower bands (ATR multiplier = 2.0)
    keltner_upper = ema20 + 2.0 * atr
    keltner_lower = ema20 - 2.0 * atr
    
    # Volume filter: volume > 1.5x 20-period MA
    volume_series = pd.Series(volume)
    volume_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma20
    
    # Weekly trend filter: price above/below weekly SMA(50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    weekly_uptrend = close > sma_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)  # ensure ATR and EMA ready
    
    for i in range(start_idx, n):
        if np.isnan(atr[i]) or np.isnan(ema20[i]) or np.isnan(volume_ma20[i]) or np.isnan(sma_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Keltner band, uptrend on weekly, volume confirmation
            if close[i] > keltner_upper[i] and weekly_uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Keltner band, downtrend on weekly, volume confirmation
            elif close[i] < keltner_lower[i] and not weekly_uptrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA(20) or weekly trend turns down
            if close[i] < ema20[i] or not weekly_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA(20) or weekly trend turns up
            if close[i] > ema20[i] or weekly_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals