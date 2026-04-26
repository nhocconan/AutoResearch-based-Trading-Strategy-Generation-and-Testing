#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_ATRStop_v2
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and ATR-based stoploss.
Tightened entry conditions by requiring volume spike AND price to close beyond the level
(avoid wicks) to reduce false breakouts. Designed for 75-150 total trades over 4 years
(19-37/year) with discrete position sizing (0.0, ±0.30) to minimize fee churn.
Works in bull/bear markets by combining price structure (Camarilla pivots) with trend
filter (1d EMA34) and volatility-based exits.
"""

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
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Load 1d data for Camarilla pivots and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla R1 = Close + (High - Low) * 1.1 / 12
    # Camarilla S1 = Close - (High - Low) * 1.1 / 12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    camarilla_R1 = close_1d + camarilla_range
    camarilla_S1 = close_1d - camarilla_range
    
    # Align Camarilla levels to LTF (1d values available after the 1d bar closes)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(14, 20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Discrete position sizing
        base_size = 0.30
        
        # Long logic: Close breaks above Camarilla R1 + price > 1d EMA34 (uptrend) + volume spike
        if close[i] > camarilla_R1_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Close breaks below Camarilla S1 + price < 1d EMA34 (downtrend) + volume spike
        elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # ATR-based stoploss: exit if price moves against position by 2.0 * ATR
        elif position == 1 and close[i] < ema_34_1d_aligned[i] - 2.0 * atr[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema_34_1d_aligned[i] + 2.0 * atr[i]:
            signals[i] = 0.0
            position = 0
        # Exit trend filter: price crosses 1d EMA34 in opposite direction
        elif position == 1 and close[i] < ema_34_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema_34_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_ATRStop_v2"
timeframe = "4h"
leverage = 1.0