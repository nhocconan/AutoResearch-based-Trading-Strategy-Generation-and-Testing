#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_weekly_bollinger_mean_reversion
# Uses weekly Bollinger Bands on 1w chart with price reversion to mean on 1d chart.
# Long when 1d close crosses below weekly BB lower band with volume confirmation.
# Short when 1d close crosses above weekly BB upper band with volume confirmation.
# Exits when price returns to weekly BB middle band (20-period SMA).
# Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag.
# Works in ranging markets via mean reversion and captures overextended moves in trends.
# Focus on BTC/ETH as primary targets.

name = "1d_1w_weekly_bollinger_mean_reversion"
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
    
    # Get weekly data for Bollinger Bands calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Bollinger Bands (20-period, 2 std dev)
    close_1w = df_1w['close'].values
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    bb_middle = sma_20  # 20-period SMA
    
    # Align weekly Bollinger Bands to daily timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1w, bb_middle)
    
    # Volume confirmation: volume > 1.3 * 20-period average (1d timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or np.isnan(bb_middle_aligned[i]):
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
        
        # Long signal: price crosses below weekly BB lower band (oversold)
        if close[i] < bb_lower_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price crosses above weekly BB upper band (overbought)
        elif close[i] > bb_upper_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to weekly BB middle band (mean reversion)
        elif position == 1 and close[i] >= bb_middle_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= bb_middle_aligned[i]:
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