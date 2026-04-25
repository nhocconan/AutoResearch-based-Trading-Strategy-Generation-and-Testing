#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_SessionFilter
Hypothesis: Trade 1h Camarilla R1/S1 breakouts with 4h EMA34 trend filter and session filter (08-20 UTC). 
Uses 4h for trend confirmation to reduce false signals in choppy 1h market. Session filter avoids low-volume 
hours that increase whipsaw. Discrete sizing 0.20 to limit fee churn. Target 15-37 trades/year on 1h.
Works in bull/bear via trend filter: long only when above 4h EMA34, short only when below.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 4h data for HTF trend (EMA34)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA34 on 4h for HTF trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla levels from previous 1h bar (for 1h entry timing)
    camarilla_range = (high - low) * 1.1 / 12.0
    camarilla_R1 = close + camarilla_range
    camarilla_S1 = close - camarilla_range
    
    # Shift by 1 to use only completed 1h bar for Camarilla calculation (no look-ahead)
    camarilla_R1 = np.roll(camarilla_R1, 1)
    camarilla_S1 = np.roll(camarilla_S1, 1)
    camarilla_R1[0] = np.nan
    camarilla_S1[0] = np.nan
    
    # Pre-compute session hours (08-20 UTC) for filter
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and Camarilla (1)
    start_idx = max(34, 1)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any data not ready
        if np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_R1[i]) or np.isnan(camarilla_S1[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + above 4h EMA34
            long_setup = (close[i] > camarilla_R1[i]) and (close[i] > ema_34_4h_aligned[i])
            # Short: price breaks below Camarilla S1 + below 4h EMA34
            short_setup = (close[i] < camarilla_S1[i]) and (close[i] < ema_34_4h_aligned[i])
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price closes below Camarilla S1 OR below 4h EMA34
            if (close[i] < camarilla_S1[i]) or (close[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price closes above Camarilla R1 OR above 4h EMA34
            if (close[i] > camarilla_R1[i]) or (close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_SessionFilter"
timeframe = "1h"
leverage = 1.0