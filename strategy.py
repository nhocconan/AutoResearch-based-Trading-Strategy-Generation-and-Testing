#!/usr/bin/env python3
# 6d_weekly_pivot_breakout_v1
# Hypothesis: Weekly pivot levels (calculated from prior week's OHLC) act as strong support/resistance. 
# Breakouts above weekly R1 or below weekly S1 with volume confirmation and 1-day trend filter capture 
# institutional breakout moves. Works in both bull and bear by following higher timeframe trend (1d EMA50).
# Target: 15-30 trades/year (60-120 total over 4 years) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "6d_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_week = get_htf_data(prices, '1w')
    if len(df_week) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    week_high = df_week['high'].values
    week_low = df_week['low'].values
    week_close = df_week['close'].values
    
    pivot = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    
    # Align weekly pivot levels to 6h
    pivot_6h = align_ltf_to_htf(prices, df_week, pivot)
    r1_6h = align_ltf_to_htf(prices, df_week, r1)
    s1_6h = align_ltf_to_htf(prices, df_week, s1)
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_ltf_to_htf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < pivot OR price < 1d EMA50
            if (close[i] < pivot_6h[i]) or (close[i] < ema_50_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > pivot OR price > 1d EMA50
            if (close[i] > pivot_6h[i]) or (close[i] > ema_50_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price > R1 + volume + price > 1d EMA50
            if (close[i] > r1_6h[i]) and volume_filter[i] and (close[i] > ema_50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < S1 + volume + price < 1d EMA50
            elif (close[i] < s1_6h[i]) and volume_filter[i] and (close[i] < ema_50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals

# Note: The function name in mtf_data is align_ltf_to_htf, not align_htf_to_ltf
# Correcting the import and usage accordingly. However, based on the rules,
# the correct function is align_htf_to_ltf. Let me fix this.

#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_v1
# Hypothesis: Weekly pivot levels (calculated from prior week's OHLC) act as strong support/resistance. 
# Breakouts above weekly R1 or below weekly S1 with volume confirmation and 1-day trend filter capture 
# institutional breakout moves. Works in both bull and bear by following higher timeframe trend (1d EMA50).
# Target: 15-30 trades/year (60-120 total over 4 years) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_week = get_htf_data(prices, '1w')
    if len(df_week) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    week_high = df_week['high'].values
    week_low = df_week['low'].values
    week_close = df_week['close'].values
    
    pivot = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    
    # Align weekly pivot levels to 6h
    pivot_6h = align_htf_to_ltf(prices, df_week, pivot)
    r1_6h = align_htf_to_ltf(prices, df_week, r1)
    s1_6h = align_htf_to_ltf(prices, df_week, s1)
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < pivot OR price < 1d EMA50
            if (close[i] < pivot_6h[i]) or (close[i] < ema_50_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > pivot OR price > 1d EMA50
            if (close[i] > pivot_6h[i]) or (close[i] > ema_50_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price > R1 + volume + price > 1d EMA50
            if (close[i] > r1_6h[i]) and volume_filter[i] and (close[i] > ema_50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < S1 + volume + price < 1d EMA50
            elif (close[i] < s1_6h[i]) and volume_filter[i] and (close[i] < ema_50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals