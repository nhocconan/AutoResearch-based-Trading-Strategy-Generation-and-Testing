#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA20_Trend
Hypothesis: Camarilla pivot points on 1h combined with 4h EMA20 trend filter and volume spike.
In bull markets: long when price breaks above R1 with uptrend and volume confirmation.
In bear markets: short when price breaks below S1 with downtrend and volume confirmation.
Uses 4h for trend direction (reduces whipsaw) and 1h for precise entry timing.
Target: 15-30 trades/year per symbol to avoid fee drag.
"""

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
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate Camarilla pivot points for 1h (using previous bar's OHLC)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    typical_price = (high + low + close) / 3.0
    range_hl = high - low
    R1 = typical_price + (1.1 * range_hl) / 12.0
    S1 = typical_price - (1.1 * range_hl) / 12.0
    
    # Shift to avoid look-ahead: use previous bar's pivot levels
    R1_prev = np.roll(R1, 1)
    S1_prev = np.roll(S1, 1)
    R1_prev[0] = np.nan
    S1_prev[0] = np.nan
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_prev[i]) or np.isnan(S1_prev[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 4h EMA20
        uptrend = close[i] > ema_20_4h_aligned[i]
        downtrend = close[i] < ema_20_4h_aligned[i]
        
        # Breakout conditions with volume confirmation
        breakout_long = close[i] > R1_prev[i] and volume_spike[i]
        breakout_short = close[i] < S1_prev[i] and volume_spike[i]
        
        # Entry conditions
        long_entry = breakout_long and uptrend
        short_entry = breakout_short and downtrend
        
        # Exit on opposite breakout (reverse position)
        long_exit = breakout_short and volume_spike[i]
        short_exit = breakout_long and volume_spike[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.20  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.20   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hEMA20_Trend"
timeframe = "1h"
leverage = 1.0