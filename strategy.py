#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Chandelier Exit with weekly trend filter and volume confirmation
# Chandelier Exit uses ATR-based trailing stops to capture trends while limiting drawdowns.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume confirmation reduces false signals. Designed for 6h timeframe with
# target of 50-150 total trades over 4 years (12-37/year). Works in bull/bear markets
# by combining trend following with volatility-based exits.
name = "6h_ChandelierExit_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_6h = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate ATR(22) for Chandelier Exit (22 periods ~ 5.5 days at 6h)
    atr_period = 22
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Chandelier Exit: 3 * ATR below highest high since entry (long) or above lowest low (short)
    # We'll calculate the trailing stop levels dynamically
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    # Volume filter: current volume > 1.3x 30-period average volume
    avg_volume = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.3 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ATR and EMA calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_6h[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_since_entry[i]) or np.isnan(lowest_since_entry[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend_up = close[i] > ema_20_6h[i]
        trend_down = close[i] < ema_20_6h[i]
        
        # Update highest/lowest since entry
        if position == 1:  # Long position
            if i == start_idx or position_prev != 1:
                highest_since_entry[i] = high[i]
            else:
                highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            chandelier_long = highest_since_entry[i] - 3.0 * atr[i]
            
            # Exit long: price closes below Chandelier Exit or trend reversal
            if close[i] < chandelier_long or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if i == start_idx or position_prev != -1:
                lowest_since_entry[i] = low[i]
            else:
                lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            chandelier_short = lowest_since_entry[i] + 3.0 * atr[i]
            
            # Exit short: price closes above Chandelier Short or trend reversal
            if close[i] > chandelier_short or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat position
            # Enter long: price closes above previous close AND trend up AND volume
            if close[i] > close[i-1] and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # Enter short: price closes below previous close AND trend down AND volume
            elif close[i] < close[i-1] and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]   # Initialize tracking
        
        position_prev = position  # Store for next iteration
    
    return signals