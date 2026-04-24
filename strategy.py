#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses 6h timeframe (primary) and 1d HTF for EMA34 trend alignment
- Camarilla levels calculated from previous completed 1d bar's OHLC (standard Camarilla formula)
- Long when price breaks above H3 AND price > 1d EMA34 (uptrend) AND volume > 2.0 * volume MA(20)
- Short when price breaks below L3 AND price < 1d EMA34 (downtrend) AND volume > 2.0 * volume MA(20)
- Exit when price reverts to the 1d VWAP (mean reversion to daily fair value)
- Discrete signal size: 0.25 to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) as per 6h timeframe recommendation
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (using previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d VWAP for exit (mean reversion target)
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_1d = (pd.Series(typical_price_1d * df_1d['volume'].values).cumsum() / 
               pd.Series(df_1d['volume'].values).cumsum()).values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate Camarilla levels from previous completed 1d bar's OHLC
    # Standard Camarilla formula: H3 = C + 1.1*(H-L)/2, L3 = C - 1.1*(H-L)/2
    # where C = close, H = high, L = low of previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    h3 = close_1d + 1.1 * camarilla_range / 2.0
    l3 = close_1d - 1.1 * camarilla_range / 2.0
    
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 1d EMA34 and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND uptrend AND volume confirmation
            if close[i] > h3_aligned[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND downtrend AND volume confirmation
            elif close[i] < l3_aligned[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to 1d VWAP
            if close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to 1d VWAP
            if close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0