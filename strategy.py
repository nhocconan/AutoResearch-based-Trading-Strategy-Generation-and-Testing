# 6h_MultiTF_Retest_Filter
# Hypothesis: Price often retests key support/resistance levels from higher timeframes after a breakout.
# Strategy: Wait for price to break above the 12h high or below the 12h low, then enter on the first retest
# of that level (within 2% tolerance) with volume confirmation. Use 1d trend (EMA50) as filter to avoid
# counter-trend trades. This captures momentum continuation while avoiding false breakouts.
# Works in both bull and bear markets by following the higher timeframe trend direction.
# Target: 50-150 total trades over 4 years.

name = "6h_MultiTF_Retest_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for structure levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 12h Structure: rolling high/low (20 periods) ---
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h rolling high and low (breakout levels)
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h levels to 6h
    high_20_12h_aligned = align_htf_to_ltf(prices, df_12h, high_20_12h)
    low_20_12h_aligned = align_htf_to_ltf(prices, df_12h, low_20_12h)
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 6h Volume Average for confirmation ---
    vol_avg_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    breakout_level = 0.0  # Track the breakout level for retest
    in_breakout_mode = False  # Track if we're waiting for retest after breakout
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(high_20_12h_aligned[i]) or np.isnan(low_20_12h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_6h[i])):
            if position != 0:
                # Simple exit: reverse signal or stop at opposite level
                if position == 1 and close_6h[i] < low_20_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] > high_20_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 6h average
        vol_confirm = volume_6h[i] > 1.5 * vol_avg_6h[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close_6h[i] > ema50_1d_aligned[i]
        downtrend = close_6h[i] < ema50_1d_aligned[i]
        
        # Check for new breakout (only when flat)
        if position == 0 and vol_confirm:
            # Bullish breakout: close above 12h rolling high
            if close_6h[i] > high_20_12h_aligned[i] and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
                breakout_level = high_20_12h_aligned[i]
                in_breakout_mode = True
            # Bearish breakout: close below 12h rolling low
            elif close_6h[i] < low_20_12h_aligned[i] and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
                breakout_level = low_20_12h_aligned[i]
                in_breakout_mode = True
        
        # Check for retest entry (when waiting for retest after breakout)
        elif position == 0 and in_breakout_mode and vol_confirm:
            # Check if price is retesting the breakout level (within 2% tolerance)
            retest_tolerance = 0.02 * breakout_level
            near_breakout_level = abs(close_6h[i] - breakout_level) <= retest_tolerance
            
            if near_breakout_level:
                # Determine direction based on breakout level type
                if breakout_level == high_20_12h_aligned[i]:
                    # Was bullish breakout, now retesting support -> go long
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_6h[i]
                else:  # breakout_level == low_20_12h_aligned[i]
                    # Was bearish breakout, now retesting resistance -> go short
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_6h[i]
                in_breakout_mode = False  # Reset after taking the retest trade
        
        # Manage existing position
        elif position != 0:
            if position == 1:
                # Long position: exit on breakdown below 12h low or trend reversal
                if close_6h[i] < low_20_12h_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                    in_breakout_mode = False
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position: exit on breakout above 12h high or trend reversal
                if close_6h[i] > high_20_12h_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    in_breakout_mode = False
                else:
                    signals[i] = -0.25
    
    return signals