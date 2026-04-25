#!/usr/bin/env python3
"""
6h Camarilla Pivot Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: Camarilla R4/S4 levels from daily pivots act as strong breakout levels.
In strong weekly trends (price > 1w EMA50 for longs, < 1w EMA50 for shorts),
breakouts at R4/S4 with volume confirmation capture institutional participation.
Works in bull markets (breakout longs in uptrend) and bear markets (breakout shorts in downtrend).
6h timeframe targets 12-37 trades/year (50-150 over 4 years) with tight entry conditions.
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
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+Close)/3 (typical price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    camarilla_width = (df_1d['high'] - df_1d['low']) * 1.1 / 2
    r4 = typical_price + camarilla_width
    s4 = typical_price - camarilla_width
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 50)  # volume MA, 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1w EMA50
        uptrend = curr_close > ema_50_aligned[i]
        downtrend = curr_close < ema_50_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Close breaks above R4 AND uptrend AND volume spike
            long_entry = (curr_close > curr_r4) and uptrend and vol_spike
            # Short: Close breaks below S4 AND downtrend AND volume spike
            short_entry = (curr_close < curr_s4) and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Close drops below R4 (breakout failure) OR loss of uptrend
            if (curr_close < curr_r4) or (curr_close < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Close rises above S4 (breakout failure) OR loss of downtrend
            if (curr_close > curr_s4) or (curr_close > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0