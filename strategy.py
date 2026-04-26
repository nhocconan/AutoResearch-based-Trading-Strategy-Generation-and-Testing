#!/usr/bin/env python3
"""
1h_Supertrend_HTF_Confluence_v1
Hypothesis: Use 1h Supertrend for entry timing with 4h/1d HTF confluence to filter noise and reduce trade frequency. 
In bull markets: 4h uptrend + 1d uptrend + 1h Supertrend long signal → long.
In bear markets: 4h downtrend + 1d downtrend + 1h Supertrend short signal → short.
In ranging markets: require both 4h and 1d to agree on trend direction to avoid whipsaw.
Position size fixed at 0.20 to manage drawdown. Target 15-30 trades/year via strict HTF alignment.
"""

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
    
    # Get 4h and 1d data for HTF trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h Supertrend calculation (ATR=10, multiplier=3.0)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_basic = hl2 + (3.0 * atr)
    lower_basic = hl2 - (3.0 * atr)
    
    # Final Upper and Lower Bands
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    for i in range(10, n):
        final_upper[i] = upper_basic[i] if (upper_basic[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]) else final_upper[i-1]
        final_lower[i] = lower_basic[i] if (lower_basic[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]) else final_lower[i-1]
    
    # Supertrend direction
    supertrend = np.full(n, np.nan)
    for i in range(10, n):
        if i == 10:
            supertrend[i] = 1 if close[i] > final_upper[i] else -1
        else:
            if supertrend[i-1] == -1 and close[i] > final_upper[i]:
                supertrend[i] = 1
            elif supertrend[i-1] == 1 and close[i] < final_lower[i]:
                supertrend[i] = -1
            else:
                supertrend[i] = supertrend[i-1]
    
    # Align Supertrend to 1h (it's already 1h, but ensure proper alignment)
    supertrend_aligned = supertrend  # Already 1h resolution
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA periods and Supertrend period
    start_idx = max(50, 50, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(supertrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        supertrend_val = supertrend_aligned[i]
        close_val = close[i]
        
        # Determine HTF trend alignment
        htf_uptrend = (close_val > ema_50_4h_val) and (close_val > ema_50_1d_val)
        htf_downtrend = (close_val < ema_50_4h_val) and (close_val < ema_50_1d_val)
        
        if position == 0:
            # Long: HTF uptrend + Supertrend long signal
            if htf_uptrend and supertrend_val == 1:
                signals[i] = 0.20
                position = 1
            # Short: HTF downtrend + Supertrend short signal
            elif htf_downtrend and supertrend_val == -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: HTF trend turns down OR Supertrend reverses
            if not htf_uptrend or supertrend_val == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: HTF trend turns up OR Supertrend reverses
            if not htf_downtrend or supertrend_val == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Supertrend_HTF_Confluence_v1"
timeframe = "1h"
leverage = 1.0