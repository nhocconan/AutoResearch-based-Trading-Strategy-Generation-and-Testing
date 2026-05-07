#!/usr/bin/env python3
name = "4h_Keltner_Channel_Trend_1dVWAP"
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Keltner Channel (20, 2) on 4h timeframe
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    upper_keltner = ema_20 + 2 * atr
    lower_keltner = ema_20 - 2 * atr
    
    # Daily VWAP (volume-weighted average price)
    vwap_1d = (df_1d['close'] * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    # Trend filter: 4h EMA(50) slope
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope = ema_50 - np.roll(ema_50, 1)
    ema_50_slope[0] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20[i]) or np.isnan(atr[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_50_slope[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above upper Keltner + price > daily VWAP + upward EMA slope
            if close[i] > upper_keltner[i] and close[i] > vwap_1d_aligned[i] and ema_50_slope[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: price below lower Keltner + price < daily VWAP + downward EMA slope
            elif close[i] < lower_keltner[i] and close[i] < vwap_1d_aligned[i] and ema_50_slope[i] < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below lower Keltner or below daily VWAP
            if close[i] < lower_keltner[i] or close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above upper Keltner or above daily VWAP
            if close[i] > upper_keltner[i] or close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Keltner Channel breakout with daily VWAP filter and trend confirmation
# - Keltner Channel (20,2) defines dynamic volatility-based support/resistance
# - Breakout above upper band with price above daily VWAP in uptrend = long signal
# - Breakdown below lower band with price below daily VWAP in downtrend = short signal
# - Daily VWAP acts as institutional reference point, effective in both bull/bear markets
# - EMA(50) slope ensures trades align with intermediate-term trend
# - Exit when price returns to lower/upper Keltner or crosses daily VWAP
# - Position size 0.25 targets ~30-50 trades/year, minimizing fee drag
# - Combines volatility breakout, volume-weighted price, and trend for robustness