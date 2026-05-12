#!/usr/bin/env python3
# 4H_VOLATILITY_BREAKOUT_VOLUME_CONFIRMATION_1D_TREND_FILTER
# Hypothesis: Volatility breakouts capture sharp directional moves, which occur in both bull and bear markets.
# Uses 10-period ATR-based breakout from close, confirmed by volume spike and 1d trend filter.
# In 1d uptrend, go long on upward volatility breakout; in downtrend, go short on downward breakout.
# Volatility breakouts tend to be fewer but higher quality trades, reducing fee drag.
# Target: 20-40 trades/year on 4h timeframe.

name = "4H_VOLATILITY_BREAKOUT_VOLUME_CONFIRMATION_1D_TREND_FILTER"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # ATR for volatility breakout (10-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros_like(close)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 9 + tr[i]) / 10  # Wilder's smoothing
    
    # Volume spike: current volume > 1.5 * 20-period average volume
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(ema34_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + upward volatility breakout + volume spike
            if (close[i] > ema34_aligned[i] and 
                close[i] > close[i-1] + 0.5 * atr[i] and  # Upward breakout
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + downward volatility breakout + volume spike
            elif (close[i] < ema34_aligned[i] and 
                  close[i] < close[i-1] - 0.5 * atr[i] and  # Downward breakout
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or volatility contraction
            if (close[i] <= ema34_aligned[i] or 
                close[i] < close[i-1] + 0.25 * atr[i]):  # Loss of upward momentum
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or volatility contraction
            if (close[i] >= ema34_aligned[i] or 
                close[i] > close[i-1] - 0.25 * atr[i]):  # Loss of downward momentum
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals