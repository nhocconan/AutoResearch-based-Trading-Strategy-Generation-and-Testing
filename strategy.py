#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_breakout_v1
# 1d timeframe with 1-week context using Camarilla pivot levels from weekly candles.
# Uses weekly high/low to calculate pivot levels for the coming week, applied on daily chart.
# Entry: daily close crosses weekly Camarilla H4/L4 with volume confirmation (>1.5x 20-day avg).
# Exit: opposite crossover or volume failure.
# Designed for low frequency (target 10-20 trades/year) to minimize fee drag.
# Works in bull markets by buying breakouts above weekly resistance, in bear markets by
# selling breakdowns below weekly support. Weekly context filters noise, volume confirms
# institutional participation.
name = "1d_1w_camarilla_breakout_v1"
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
    
    # Get weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week
    high_prev = df_1w['high'].shift(1).values
    low_prev = df_1w['low'].shift(1).values
    close_prev = df_1w['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_h4 = close_prev + range_prev * 1.1 / 2
    camarilla_l4 = close_prev - range_prev * 1.1 / 2
    
    # Align to 1d timeframe (weekly levels remain constant until next weekly bar)
    h4_level = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    l4_level = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if levels not ready
        if np.isnan(h4_level[i]) or np.isnan(l4_level[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new signals
        if vol_confirm[i]:
            # Long signal: price breaks above H4
            if close[i] > h4_level[i] and position != 1:
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below L4
            elif close[i] < l4_level[i] and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit on opposite crossover
            elif close[i] < l4_level[i] and position == 1:
                position = 0
                signals[i] = 0.0
            elif close[i] > h4_level[i] and position == -1:
                position = 0
                signals[i] = 0.0
            # Hold position if no crossover
            else:
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # No volume confirmation: hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals