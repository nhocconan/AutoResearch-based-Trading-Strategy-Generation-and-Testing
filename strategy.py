#!/usr/bin/env python3
# Hypothesis: 1h EMA(20) pullback strategy with 4h trend filter (EMA50) and session filter (08-20 UTC).
# Long when: price > 4h EMA50 (bullish trend), price pulls back to touch 1h EMA20 from above, and session is active.
# Short when: price < 4h EMA50 (bearish trend), price pulls back to touch 1h EMA20 from below, and session is active.
# Exit on opposite 1h EMA20 touch or session end.
# Uses 4h for trend direction (reduces whipsaw), 1h for precise entry timing, and session filter to avoid low-volatility periods.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.

name = "1h_EMA20_Pullback_4hEMA50_Trend_Session"
timeframe = "1h"
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = prices.index.hour  # open_time is already datetime64[ms], index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # --- 1h Indicators (LTF) ---
    # 1h EMA(20) for pullback entries
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) - trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after EMA20 warmup
        # Skip if missing data or outside session
        if (np.isnan(ema_20[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: bullish trend (price > 4h EMA50) + pullback to 1h EMA20 from above
            if (close[i] > ema_50_4h_aligned[i] and 
                low[i] <= ema_20[i] and 
                close[i-1] > ema_20[i-1]):  # came from above
                signals[i] = 0.20
                position = 1
            # SHORT: bearish trend (price < 4h EMA50) + pullback to 1h EMA20 from below
            elif (close[i] < ema_50_4h_aligned[i] and 
                  high[i] >= ema_20[i] and 
                  close[i-1] < ema_20[i-1]):  # came from below
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below 1h EMA20 or session ends next bar
            if (low[i] < ema_20[i]) or (i+1 < n and not in_session[i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price breaks above 1h EMA20 or session ends next bar
            if (high[i] > ema_20[i]) or (i+1 < n and not in_session[i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals