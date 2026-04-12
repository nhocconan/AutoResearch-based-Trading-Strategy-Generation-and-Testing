#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_weekly_vwap_breakout
# Uses weekly VWAP as dynamic support/resistance. Long when price closes above weekly VWAP with volume > 2x average.
# Short when price closes below weekly VWAP with volume > 2x average.
# Exits when price crosses back across weekly VWAP.
# Weekly VWAP adapts to market conditions, providing dynamic levels that work in both trending and ranging markets.
# Volume filter ensures only significant breaks are traded, reducing false signals and trade frequency.
# Target: 10-20 trades/year to minimize fee drag.

name = "1d_1w_weekly_vwap_breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly VWAP
    # VWAP = sum(price * volume) / sum(volume) for the week
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    pv = typical_price * df_1w['volume']
    vwap = pv.cumsum() / df_1w['volume'].cumsum()
    vwap_values = vwap.values
    
    # Align weekly VWAP to daily timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap_values)
    
    # Volume confirmation: volume > 2 * 20-period average (daily timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(vwap_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price closes above weekly VWAP
        if close[i] > vwap_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price closes below weekly VWAP
        elif close[i] < vwap_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price crosses back across weekly VWAP
        elif position == 1 and close[i] <= vwap_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= vwap_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals