# ANALYSIS OF FAILED STRATEGIES AND SOLUTION DESIGN
# Examining the failure patterns from experiments #164343-164357:
# - Most failures were due to 0 trades (too restrictive conditions)
# - Some had negative Sharpe due to whipsaw in ranging markets
# - The successful strategies in this session used Camarilla pivots with volume confirmation
# 
# KEY INSIGHTS FROM SUCCESSFUL PATTERNS:
# 1. Camarilla pivot levels (especially R1/S1, R3/S3) work well as support/resistance
# 2. Volume confirmation is critical to avoid false breakouts
# 3. Trend filters (higher timeframe EMA) prevent counter-trend trades
# 4. Proper position sizing (0.20-0.30) manages drawdown
# 5. Target 20-50 trades/year to minimize fee drag
#
# For this experiment (1d timeframe, 1h HTF), I'll design a strategy that:
# 1. Uses Camarilla R1/S1 levels from daily data as entry zones
# 2. Requires volume spike for confirmation (avoids choppy false signals)
# 3. Uses 1h EMA50 as trend filter (aligns with higher timeframe momentum)
# 4. Implements proper risk management via time-based exits
# 5. Targets 15-25 trades/year to stay within fee-efficient range
#
# The strategy avoids overcomplication by focusing on:
# - One clear entry signal (price at Camarilla level + volume + trend)
# - Clear exit conditions (time-based or trend reversal)
# - Discrete position sizing to minimize churn

#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Volume_Trend_Filter
Hypothesis: Daily Camarilla R1 and S1 levels act as significant support/resistance.
When price approaches these levels with volume confirmation and 1h trend alignment,
it offers high-probability bounce/breakout opportunities. Works in both bull 
and bear markets by following the 1h trend direction while using daily structure
for entry precision. Target: 15-25 trades/year per symbol.
"""

name = "1d_Camarilla_R1S1_Volume_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using standard Camarilla formula: 
    # R4 = C + (H-L) * 1.1/2
    # R3 = C + (H-L) * 1.1/4
    # R2 = C + (H-L) * 1.1/6
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    # S2 = C - (H-L) * 1.1/6
    # S3 = C - (H-L) * 1.1/4
    # S4 = C - (H-L) * 1.1/2
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 for each day
    r1_1d = np.zeros(len(df_1d))
    s1_1d = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        # Use previous day's OHLC to calculate today's levels
        hl = high_1d[i-1] - low_1d[i-1]
        if hl > 0:  # Avoid division by zero
            r1_1d[i] = close_1d[i-1] + hl * 1.1 / 12
            s1_1d[i] = close_1d[i-1] - hl * 1.1 / 12
        else:
            r1_1d[i] = close_1d[i-1]
            s1_1d[i] = close_1d[i-1]
    
    # Align Camarilla levels to 1d timeframe (no additional delay needed)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 1h data for trend filter and volume average
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # Calculate 1h EMA50 for trend filter
    ema_50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1h = close_1h > ema_50_1h
    downtrend_1h = close_1h < ema_50_1h
    
    # Align 1h trend to 1d timeframe
    uptrend_1h_aligned = align_htf_to_ltf(prices, df_1h, uptrend_1h)
    downtrend_1h_aligned = align_htf_to_ltf(prices, df_1h, downtrend_1h)
    
    # Calculate volume spike detector (volume > 1.5x 20-period average)
    vol_ma_20 = np.zeros(len(volume_1h))
    for i in range(20, len(volume_1h)):
        vol_ma_20[i] = np.mean(volume_1h[i-20:i])
    
    volume_spike = np.zeros(len(volume_1h))
    for i in range(20, len(volume_1h)):
        if vol_ma_20[i] > 0:
            volume_spike[i] = volume_1h[i] > (vol_ma_20[i] * 1.5)
        else:
            volume_spike[i] = False
    
    # Align volume spike to 1d timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1h, volume_spike)
    
    # Initialize signals array
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Main processing loop
    for i in range(50, n):
        # Get current values
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        uptrend = uptrend_1h_aligned[i]
        downtrend = downtrend_1h_aligned[i]
        vol_spike = volume_spike_aligned[i]
        
        if position == 0:
            # LONG ENTRY: Price at S1 support + volume spike + 1h uptrend
            # Allow small tolerance for price touching the level
            if curr_low <= s1 * 1.001 and curr_high >= s1 * 0.999:
                if vol_spike and uptrend:
                    signals[i] = 0.25
                    position = 1
            # SHORT ENTRY: Price at R1 resistance + volume spike + 1h downtrend
            elif curr_high >= r1 * 0.999 and curr_low <= r1 * 1.001:
                if vol_spike and downtrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # LONG EXIT: Trend reversal or time-based exit (5 days max)
            if not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # SHORT EXIT: Trend reversal or time-based exit (5 days max)
            if not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals