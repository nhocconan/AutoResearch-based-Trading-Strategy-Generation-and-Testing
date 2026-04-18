#3:00 PM
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot H3/L3 breakout with weekly trend filter and volume confirmation.
# Camarilla levels provide statistically significant support/resistance based on prior day's range.
# Weekly EMA34 filter ensures trades align with higher timeframe trend to avoid counter-trend whipsaws.
# Volume confirmation adds conviction to breakouts.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 12h timeframe.
# Works in bull markets (breakouts above H3) and bear markets (breakouts below L3).
name = "12h_Camarilla_H3L3_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Get daily data for Camarilla pivot levels (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly EMA34 for trend filter
    close_w = df_1w['close'].values
    ema34_w = pd.Series(close_w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily Camarilla pivot levels (H3, L3)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Pivot point calculation
    pivot = (high_d + low_d + close_d) / 3.0
    range_d = high_d - low_d
    
    # Camarilla levels: H3 = close + range * 1.1/4, L3 = close - range * 1.1/4
    h3 = close_d + (range_d * 1.1 / 4)
    l3 = close_d - (range_d * 1.1 / 4)
    
    # Align weekly trend and daily Camarilla levels to 12h timeframe
    ema34_w_aligned = align_htf_to_ltf(prices, df_1w, ema34_w)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate 20-period average volume for confirmation (using 12h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_w_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above H3 AND weekly uptrend (price > EMA34) AND volume confirmation
            long_breakout = close[i] > h3_aligned[i]
            uptrend = close[i] > ema34_w_aligned[i]
            if vol_confirm and uptrend and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND weekly downtrend (price < EMA34) AND volume confirmation
            elif vol_confirm and (close[i] < ema34_w_aligned[i]) and (close[i] < l3_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below L3 OR weekly trend turns down
            exit_condition = close[i] < l3_aligned[i] or close[i] < ema34_w_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above H3 OR weekly trend turns up
            exit_condition = close[i] > h3_aligned[i] or close[i] > ema34_w_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals