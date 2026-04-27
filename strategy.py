#!/usr/bin/env python3
"""
12h_Camarilla_P1_P2_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels P1 (H5) and P2 (H6) from daily data act as strong resistance/support. Price breaking above P1 with 1d EMA34 uptrend and volume spike captures bullish moves; breaking below P2 with 1d EMA34 downtrend and volume spike captures bearish moves. Works in bull via P1 breakouts and bear via P2 breakdowns. Targets 15-25 trades/year on 12h to minimize fee drag.
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: H5 = C + (H-L)*1.1/2, H6 = C + (H-L)*1.1
    # These are the outer resistance/support levels (P1 and P2 in some naming)
    camarilla_range = (high_1d - low_1d) * 1.1
    p1_1d = close_1d + camarilla_range * 0.5  # H5 level
    p2_1d = close_1d + camarilla_range        # H6 level
    
    # Align P1/P2 to 12h timeframe (use previous day's levels)
    p1_1d_aligned = align_htf_to_ltf(prices, df_1d, p1_1d)
    p2_1d_aligned = align_htf_to_ltf(prices, df_1d, p2_1d)
    
    # Get 1d data for trend filter (EMA34)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2.0 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(34, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(p1_1d_aligned[i]) or np.isnan(p2_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        p1 = p1_1d_aligned[i]
        p2 = p2_1d_aligned[i]
        ema_trend = ema34_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above P1 with uptrend and volume spike
            if close[i] > p1 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below P2 with downtrend and volume spike
            elif close[i] < p2 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below P2 or trend turns down
            if close[i] < p2 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above P1 or trend turns up
            if close[i] > p1 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_P1_P2_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0