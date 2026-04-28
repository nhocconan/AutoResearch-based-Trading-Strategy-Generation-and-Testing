#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels (R3/S3) for mean reversion and (R4/S4) for breakout continuation.
# Long: price crosses above S3 with close > weekly EMA34 (uptrend filter) and volume > 1.5x 20-bar average.
# Short: price crosses below R3 with close < weekly EMA34 (downtrend filter) and volume > 1.5x 20-bar average.
# Breakout continuation: long when price breaks above R4 with volume confirmation; short when breaks below S4.
# Uses discrete position sizing (0.25) to limit drawdown. Weekly Camarilla provides structure from higher timeframe,
# volume confirms momentum, and weekly EMA34 filters counter-trend noise. Works in bull (breakouts via R4/S4) and bear
# (mean reversion at R3/S3) markets via regime-adaptive logic.

name = "6h_WeeklyCamarilla_R3S3_R4S4_MeanRev_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots and EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels based on prior week's OHLC
    # Camarilla: H = high, L = low, C = close of prior week
    H = df_1w['high'].shift(1).values  # prior week high
    L = df_1w['low'].shift(1).values   # prior week low
    C = df_1w['close'].shift(1).values # prior week close
    
    # Calculate Camarilla levels for prior week
    R4 = C + (H - L) * 1.1 / 2
    R3 = C + (H - L) * 1.1 / 4
    S3 = C - (H - L) * 1.1 / 4
    S4 = C - (H - L) * 1.1 / 2
    
    # Align weekly Camarilla levels to 6h timeframe (wait for weekly close)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 6h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion at R3/S3 with trend filter and volume confirmation
        long_mean_rev = (close[i-1] <= S3_aligned[i-1] and close[i] > S3_aligned[i]) and \
                        close[i] > ema_34_1w_aligned[i] and volume_confirm[i]
        short_mean_rev = (close[i-1] >= R3_aligned[i-1] and close[i] < R3_aligned[i]) and \
                         close[i] < ema_34_1w_aligned[i] and volume_confirm[i]
        
        # Breakout continuation at R4/S4 with volume confirmation
        long_breakout = close[i] > R4_aligned[i] and volume_confirm[i]
        short_breakout = close[i] < S4_aligned[i] and volume_confirm[i]
        
        # Exit conditions: opposite Camarilla level (R3/S3) or opposite breakout level (S4/R4)
        long_exit = close[i] < R3_aligned[i] or close[i] > S4_aligned[i]
        short_exit = close[i] > S3_aligned[i] or close[i] < R4_aligned[i]
        
        # Handle entries and exits
        if (long_mean_rev or long_breakout) and position <= 0:
            signals[i] = 0.25
            position = 1
        elif (short_mean_rev or short_breakout) and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals