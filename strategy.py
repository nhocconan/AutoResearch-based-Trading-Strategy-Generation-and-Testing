# 1h_PivotBreak_4hTrend_Volume
# Hypothesis: On 1h timeframe, breakout above/below 4h pivot (HL2) with volume spike
# follows 4h trend direction. This gives breakout precision while keeping trade
# frequency low (15-35/year) by requiring 4h trend alignment. Works in bull
# (breakouts with uptrend) and bear (breakdowns with downtrend). Uses discrete
# size 0.20 to minimize churn. Target: 60-150 total trades over 4 years.

name = "1h_PivotBreak_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h trend: EMA34 on close
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # 4h pivot (HL2) for breakout levels
    hl2_4h = (df_4h['high'].values + df_4h['low'].values) / 2
    hl2_4h_aligned = align_htf_to_ltf(prices, df_4h, hl2_4h)
    
    # Volume spike: > 1.6x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > 1.6 * vol_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Wait for volume MA and EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema34_4h_aligned[i]) or np.isnan(hl2_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 4h pivot with volume spike in 4h uptrend
            if close[i] > hl2_4h_aligned[i] and vol_spike[i] and close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below 4h pivot with volume spike in 4h downtrend
            elif close[i] < hl2_4h_aligned[i] and vol_spike[i] and close[i] < ema34_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: close below 4h pivot or trend turns down
            if close[i] < hl2_4h_aligned[i] or close[i] < ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: close above 4h pivot or trend turns up
            if close[i] > hl2_4h_aligned[i] or close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

#!/usr/bin/env python3