#!/usr/bin/env python3
name = "1d_1w_Trend_Continuation_With_Volume_Confirmation"
timeframe = "1d"
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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter (only trade in direction of weekly trend)
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 1d ATR(14) for volatility filter and stoploss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = np.zeros_like(tr)
    atr[14] = np.mean(tr[1:15])
    for i in range(15, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # 1d volume spike: > 2x 20-period average
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume > 2.0 * vol_ma_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above weekly EMA34 with volume spike
            if close[i] > ema34_1w_aligned[i] and vol_spike_1d[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below weekly EMA34 with volume spike
            elif close[i] < ema34_1w_aligned[i] and vol_spike_1d[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below weekly EMA34 or ATR-based stop (2x ATR from entry)
            if close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above weekly EMA34 or ATR-based stop
            if close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly trend provides directional bias, daily volume confirms momentum.
# Works in bull markets (ride the trend) and bear markets (short the trend).
# Volume spike filters out low-momentum moves. ATR-based exit manages risk.
# Target: 10-25 trades/year to minimize fee drag. Position size 0.25 limits drawdown.