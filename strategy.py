# 1h_4D_Trend_Filter_v1: 1h entries with 4h/1d trend confirmation
# Uses 4h EMA20 for trend, 1d EMA50 for trend filter, and volume spike on 1h
# Entry: price crosses EMA20(4h) in direction of EMA50(1d) with volume confirmation
# Exit: price crosses back or trend changes
# Target: 15-37 trades/year per symbol, total 60-150 over 4 years
# Session filter: 08-20 UTC to avoid low liquidity periods
# Position size: 0.20 (20% of capital)

#!/usr/bin/env python3
name = "1h_4D_Trend_Filter_v1"
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
    
    # === 4H DATA FOR TREND DIRECTION ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # EMA20 on 4h for trend direction
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # === 1D DATA FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on 1d for trend filter (stronger filter)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === VOLUME CONFIRMATION (1h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)  # Moderate volume spike
    
    # === SESSION FILTER (08-20 UTC) ===
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price above EMA20(4h) AND above EMA50(1d) + volume confirmation
            if (close[i] > ema20_4h_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: price below EMA20(4h) AND below EMA50(1d) + volume confirmation
            elif (close[i] < ema20_4h_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: price crosses below EMA20(4h) OR below EMA50(1d)
            if close[i] < ema20_4h_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price crosses above EMA20(4h) OR above EMA50(1d)
            if close[i] > ema20_4h_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals