#!/usr/bin/env python3
# Hypothesis: 1h EMA(21) pullback strategy with 4h Supertrend(10,3) filter and session (08-20 UTC) for entry timing.
# Long when: price > 4h Supertrend (uptrend), price pulls back to touch 1h EMA21 from above, and session filter active.
# Short when: price < 4h Supertrend (downtrend), price pulls back to touch 1h EMA21 from below, and session filter active.
# Uses 4h for trend direction, 1h for precise entry timing, session filter to reduce noise.
# Target: 60-150 trades over 4 years (15-37/year) by requiring confluence of trend, pullback, and session.

name = "1h_EMA21_Pullback_4hSupertrend_Session_v1"
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
    volume = prices['volume'].values
    
    # Calculate ATR(10) for Supertrend
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Calculate Supertrend on 4h timeframe
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # ATR for Supertrend calculation (using 4h data)
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Supertrend basic upper and lower bands
    hl2_4h = (high_4h + low_4h) / 2
    upper_band_4h = hl2_4h + (3.0 * atr_4h)
    lower_band_4h = hl2_4h - (3.0 * atr_4h)
    
    # Initialize Supertrend arrays
    supertrend_4h = np.full_like(close_4h, np.nan)
    direction_4h = np.full_like(close_4h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(len(close_4h)):
        if i == 0:
            supertrend_4h[i] = hl2_4h[i]
            direction_4h[i] = 1
        else:
            if close_4h[i-1] > supertrend_4h[i-1]:
                # Previous close was above previous Supertrend
                supertrend_4h[i] = max(lower_band_4h[i], supertrend_4h[i-1])
                direction_4h[i] = 1
            else:
                # Previous close was below previous Supertrend
                supertrend_4h[i] = min(upper_band_4h[i], supertrend_4h[i-1])
                direction_4h[i] = -1
    
    # Align Supertrend and direction to 1h timeframe (wait for completed 4h bar)
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # Calculate EMA21 on 1h close
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours from index)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):  # Start after EMA21 warmup
        # Skip if any required data is NaN
        if (np.isnan(supertrend_4h_aligned[i]) or np.isnan(direction_4h_aligned[i]) or 
            np.isnan(ema21[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 4h uptrend, price touches EMA21 from above, in session
            if (direction_4h_aligned[i] == 1 and 
                low[i] <= ema21[i] <= high[i] and 
                close[i] > ema21[i] and  # Confirmed bounce from EMA21
                in_session[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: 4h downtrend, price touches EMA21 from below, in session
            elif (direction_4h_aligned[i] == -1 and 
                  low[i] <= ema21[i] <= high[i] and 
                  close[i] < ema21[i] and  # Confirmed rejection from EMA21
                  in_session[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price closes below EMA21 or 4h trend turns down
            if close[i] < ema21[i] or direction_4h_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price closes above EMA21 or 4h trend turns up
            if close[i] > ema21[i] or direction_4h_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals