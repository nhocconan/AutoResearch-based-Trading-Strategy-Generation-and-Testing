#!/usr/bin/env python3
# 1h_Keltner_MeanReversion_4hTrend
# Hypothesis: In 1h timeframe, price often reverts to the mean during range-bound periods, but only when aligned with the 4h trend. 
# Uses Keltner Channel (20, 1.5) for mean reversion signals and 4h EMA50 for trend filter. 
# Long when price touches lower KC in 4h uptrend, short when price touches upper KC in 4h downtrend. 
# Volume filter (>1.5x 20-period MA) confirms momentum. 
# Session filter (08-20 UTC) reduces noise. 
# Target: 15-30 trades/year per symbol via tight entry conditions.

name = "1h_Keltner_MeanReversion_4hTrend"
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
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h Keltner Channel (20, 1.5)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean()
    atr = pd.Series(np.abs(high - low)).rolling(window=20, min_periods=20).mean()
    upper_kc = ema_20 + 1.5 * atr
    lower_kc = ema_20 - 1.5 * atr
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50_4h (50), EMA20 (20), ATR (20), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(upper_kc[i]) or 
            np.isnan(lower_kc[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 4h trend filter
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: 4h uptrend + price touches lower KC + volume + session
            if uptrend_4h and close[i] <= lower_kc[i] and volume_confirm and in_session[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: 4h downtrend + price touches upper KC + volume + session
            elif downtrend_4h and close[i] >= upper_kc[i] and volume_confirm and in_session[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h trend breaks or price crosses above EMA20
            if not uptrend_4h or close[i] >= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h trend breaks or price crosses below EMA20
            if not downtrend_4h or close[i] <= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals