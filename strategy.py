#!/usr/bin/env python3
name = "1h_Camarilla_R3_S4_Breakout_4hEMA50_VolumeSpike_Session"
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
    
    # === 4H DATA FOR EMA50 TREND FILTER ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # === CALCULATE EMA50 FOR TREND FILTER ===
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === CALCULATE CAMARILLA LEVELS (R3, S4) FROM PREVIOUS DAY ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    rng = high_1d - low_1d
    R3 = close_1d + rng * 1.1 / 4
    S4 = close_1d - rng * 1.1 / 4
    
    # ALIGN TO 1H TIMEFRAME (PREVIOUS DAY'S LEVELS AVAILABLE AT OPEN)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    R3_1h = align_htf_to_ltf(prices, df_1d, R3)
    S4_1h = align_htf_to_ltf(prices, df_1d, S4)
    
    # === VOLUME SPIKE DETECTION (20-PERIOD) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # === SESSION FILTER: 08-20 UTC ===
    # Pre-compute hour array for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(R3_1h[i]) or
            np.isnan(S4_1h[i]) or
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: PRICE BREAKS ABOVE R3 + ABOVE 4H EMA50 + VOLUME SPIKE + IN SESSION
            if (close[i] > R3_1h[i] and 
                close[i] > ema50_4h_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: PRICE BREAKS BELOW S4 + BELOW 4H EMA50 + VOLUME SPIKE + IN SESSION
            elif (close[i] < S4_1h[i] and 
                  close[i] < ema50_4h_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: PRICE BREAKS BELOW S4 (REVERSAL) OR BELOW 4H EMA50
            if close[i] < S4_1h[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: PRICE BREAKS ABOVE R3 (REVERSAL) OR ABOVE 4H EMA50
            if close[i] > R3_1h[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals