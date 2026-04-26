#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Filter_v1
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter on 12h timeframe.
Long when price breaks above R1 (resistance) in uptrend (close > EMA34).
Short when price breaks below S1 (support) in downtrend (close < EMA34).
Uses discrete sizing 0.25 to minimize fee churn. Designed to work in both bull and bear markets
by following the 1d trend while using Camarilla levels for precise entry/exit.
Target trades: 12-37/year (50-150 total over 4 years) to stay within fee drag limits.
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
    
    # Get 1d data for Camarilla pivot calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    tp_h = typical_price['high'].values if hasattr(typical_price, 'high') else typical_price.values
    tp_l = typical_price['low'].values if hasattr(typical_price, 'low') else typical_price.values
    tp_c = typical_price['close'].values if hasattr(typical_price, 'close') else typical_price.values
    
    # Actually, typical_price is a Series, so:
    tp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    tp_h = df_1d['high'].values
    tp_l = df_1d['low'].values
    tp_c = df_1d['close'].values
    tp = (tp_h + tp_l + tp_c) / 3
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where H,L,C are from previous day
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_range = (h_1d - l_1d) * 1.1 / 12
    r_1d = c_1d + camarilla_range  # R1
    s_1d = c_1d - camarilla_range  # S1
    
    # Align Camarilla levels (no extra delay needed as they're based on completed 1d bar)
    r_1d_aligned = align_htf_to_ltf(prices, df_1d, r_1d)
    s_1d_aligned = align_htf_to_ltf(prices, df_1d, s_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(c_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 and at least one 1d bar for Camarilla
    start_idx = max(34, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r_1d_aligned[i]) or 
            np.isnan(s_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        r_1d_val = r_1d_aligned[i]
        s_1d_val = s_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        if position == 0:
            # Long: price breaks above R1 (resistance) in uptrend
            long_signal = (high_val > r_1d_val) and (close_val > ema_34_1d_val)
            # Short: price breaks below S1 (support) in downtrend
            short_signal = (low_val < s_1d_val) and (close_val < ema_34_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below S1 (support) or trend reversal
            if close_val < s_1d_val or close_val < ema_34_1d_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above R1 (resistance) or trend reversal
            if close_val > r_1d_val or close_val > ema_34_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Filter_v1"
timeframe = "12h"
leverage = 1.0