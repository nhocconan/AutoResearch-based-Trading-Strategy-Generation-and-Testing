#!/usr/bin/env python3
# 4h_12h_keltner_breakout_volume_v1
# Strategy: 4h Keltner Channel breakout with volume confirmation and 12h EMA trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Keltner Channels capture volatility-based breakouts. In bull markets, price breaks above upper band with volume and 12h EMA uptrend. In bear markets, price breaks below lower band with volume and 12h EMA downtrend. Volume confirms breakout sincerity. Low trade frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_keltner_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h Keltner Channel (20, 2.0)
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    atr_series = pd.Series(high - low).rolling(window=20, min_periods=20).mean()  # Simplified ATR
    kc_middle = tp_series.rolling(window=20, min_periods=20).mean()
    kc_upper = kc_middle + (2.0 * atr_series)
    kc_lower = kc_middle - (2.0 * atr_series)
    kc_upper = kc_upper.values
    kc_lower = kc_lower.values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.8 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Entry logic: Keltner breakout + volume + trend alignment
        if close[i] > kc_upper[i] and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] < kc_lower[i] and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to middle band
        elif position == 1 and close[i] < kc_middle.iloc[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > kc_middle.iloc[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals