#!/usr/bin/env python3
# 12h_1w_keltner_breakout_v1
# Strategy: 12h Keltner breakout with 1w EMA trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price breaking above/below Keltner Channel (ATR-based) with strong volume captures breakout moves.
#             1-week EMA filter ensures alignment with higher timeframe trend, reducing false breakouts in chop.
#             Works in bull via upside breaks in uptrend, in bear via downside breaks in downtrend.
#             Low trade frequency (~15-30/year) via strict breakout conditions minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_keltner_breakout_v1"
timeframe = "12h"
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
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 12h ATR(20) for Keltner Channel
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar: no previous close
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 12h EMA(20) for Keltner Channel middle
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bounds
    keltner_upper = ema_20 + (2.0 * atr)
    keltner_lower = ema_20 - (2.0 * atr)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1w EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry logic: Keltner breakout + volume + trend alignment
        if close[i] > keltner_upper[i] and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] < keltner_lower[i] and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price crosses back inside Keltner Channel
        elif position == 1 and close[i] < keltner_lower[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > keltner_upper[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals