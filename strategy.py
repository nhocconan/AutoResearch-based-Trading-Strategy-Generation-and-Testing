#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Supertrend for trend direction and 1d ATR-based volatility breakout.
# Supertrend(12h, ATR=10, mult=3.0) filters trend direction to avoid counter-trend trades.
# Entry: price breaks above/below 1d ATR(14) from session open with volume confirmation (>1.5x 20-bar avg).
# Exit: opposite ATR break or Supertrend reversal.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Target: 100-180 total trades over 4 years = 25-45/year for 4h (within proven winning range).

name = "4h_12hSupertrend_1dATRBreakout_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First element NaN
    
    # ATR(10)
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_12h + low_12h) / 2
    upper_basic = hl2 + 3.0 * atr_10
    lower_basic = hl2 - 3.0 * atr_10
    
    # Final Upper and Lower Bands
    final_upper = np.full_like(close_12h, np.nan)
    final_lower = np.full_like(close_12h, np.nan)
    
    for i in range(1, len(close_12h)):
        if close_12h[i-1] <= final_upper[i-1]:
            final_upper[i] = min(upper_basic[i], final_upper[i-1])
        else:
            final_upper[i] = upper_basic[i]
            
        if close_12h[i-1] >= final_lower[i-1]:
            final_lower[i] = max(lower_basic[i], final_lower[i-1])
        else:
            final_lower[i] = lower_basic[i]
    
    # Supertrend
    supertrend = np.full_like(close_12h, np.nan)
    for i in range(1, len(close_12h)):
        if supertrend[i-1] == final_upper[i-1]:
            supertrend[i] = final_lower[i] if close_12h[i] <= final_lower[i] else final_upper[i]
        else:
            supertrend[i] = final_upper[i] if close_12h[i] >= final_upper[i] else final_lower[i]
    
    # Align Supertrend to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    
    # Calculate 1d ATR(14) for volatility breakout
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # ATR(14)
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ATR to 4h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate session open price for each 4h bar (using 1d open)
    open_1d = df_1d['open'].values
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, open_1d)
    
    # Calculate 4h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(open_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from Supertrend
        uptrend = close[i] > supertrend_aligned[i]
        downtrend = close[i] < supertrend_aligned[i]
        
        # ATR breakout conditions with volume confirmation
        long_breakout = close[i] > open_1d_aligned[i] + atr_14_1d_aligned[i] and volume_confirm[i]
        short_breakout = close[i] < open_1d_aligned[i] - atr_14_1d_aligned[i] and volume_confirm[i]
        
        # Exit conditions: opposite ATR break or Supertrend reversal
        long_exit = close[i] < open_1d_aligned[i] - atr_14_1d_aligned[i] or not uptrend
        short_exit = close[i] > open_1d_aligned[i] + atr_14_1d_aligned[i] or not downtrend
        
        # Handle entries and exits
        if long_breakout and uptrend and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and downtrend and position >= 0:
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