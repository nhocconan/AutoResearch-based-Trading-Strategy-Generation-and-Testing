# 1h_4d_vwap_breakout_v1
# Uses VWAP from 4-hour timeframe with volume confirmation and session filter.
# Long when price crosses above VWAP with above-average volume during active session.
# Short when price crosses below VWAP with above-average volume during active session.
# VWAP acts as dynamic support/resistance, volume confirms institutional interest.
# Session filter (08-20 UTC) reduces noise from low-liquidity periods.
# Target: 20-40 trades/year per symbol for low friction and high edge.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_vwap_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for VWAP calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate typical price and VWAP for 4h
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    vwap_numerator = (typical_price * df_4h['volume']).cumsum()
    vwap_denominator = df_4h['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    
    # Align VWAP to 1h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_4h, vwap.values)
    
    # Volume confirmation: volume > 1.3 * 20-period average (1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if VWAP not ready or not in session
        if np.isnan(vwap_aligned[i]) or not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Check volume filter
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price crosses above VWAP with volume
        if close[i] > vwap_aligned[i] and close[i-1] <= vwap_aligned[i-1] and position != 1:
            position = 1
            signals[i] = 0.20
        # Short signal: price crosses below VWAP with volume
        elif close[i] < vwap_aligned[i] and close[i-1] >= vwap_aligned[i-1] and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit conditions: opposite cross
        elif close[i] < vwap_aligned[i] and close[i-1] >= vwap_aligned[i-1] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > vwap_aligned[i] and close[i-1] <= vwap_aligned[i-1] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals