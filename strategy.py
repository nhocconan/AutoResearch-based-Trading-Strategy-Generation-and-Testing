#!/usr/bin/env python3
# Hypothesis: 1h momentum strategy with 4h trend filter and volume confirmation.
# Uses 4h EMA50 for trend direction, enters on 1h momentum bursts (price > 1h EMA20 + volume spike)
# only during active sessions (08-20 UTC) to avoid low-volume noise.
# Designed for moderate trade frequency (~20-40/year) to balance opportunity and fee cost.

name = "1h_Momentum_4hTrend_Volume_Session"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h EMA20 for momentum filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 30-period average
    volume_series = pd.Series(volume)
    vol_ma30 = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_ok = volume > vol_ma30
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # Already datetime64[ms], .hour works
    session_ok = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(ema20[i]) or np.isnan(vol_ma30[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above 1h EMA20, 4h uptrend, volume spike, active session
            if (close[i] > ema20[i] and 
                close[i] > ema50_4h_aligned[i] and 
                volume_ok[i] and 
                session_ok[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price below 1h EMA20, 4h downtrend, volume spike, active session
            elif (close[i] < ema20[i] and 
                  close[i] < ema50_4h_aligned[i] and 
                  volume_ok[i] and 
                  session_ok[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1h EMA20 or 4h trend turns down
            if close[i] < ema20[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above 1h EMA20 or 4h trend turns up
            if close[i] > ema20[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals