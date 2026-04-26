#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dEMAFilter_v1
Hypothesis: On 1h timeframe, trade long when price breaks above Camarilla R1 level and short when breaks below S1 level,
filtered by 4h EMA20 trend and 1d EMA50 regime filter. Camarilla levels provide intraday support/resistance derived from 
prior day's range. 4h EMA20 acts as medium-timeframe trend filter. 1d EMA50 acts as higher-timeframe regime filter to avoid 
counter-trend trades in strong bear/bull markets. Designed for 60-150 total trades over 4 years (15-37/year) with discrete 
sizing (0.20) to minimize fee drag. Uses session filter (08-20 UTC) to reduce noise trades outside active market hours.
Works in bull/bear markets via 4h/1d trend filters and volatility-based stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC) once before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA20 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate Camarilla levels from prior 4h bar (prior day's range equivalent for 4h)
    # For 4h timeframe, use prior 4h bar's high/low to compute intraday levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True range for prior 4h bar
    prev_close_4h = np.roll(close_4h, 1)
    prev_close_4h[0] = close_4h[0]  # first bar
    tr_4h = np.maximum(high_4h - low_4h, np.maximum(np.abs(high_4h - prev_close_4h), np.abs(low_4h - prev_close_4h)))
    atr_4h = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values  # Wilder's ATR
    
    # Camarilla levels: based on prior 4h bar's range
    hl_range_4h = high_4h - low_4h
    r1_4h = close_4h + 0.5 * hl_range_4h
    s1_4h = close_4h - 0.5 * hl_range_4h
    
    # Align HTF indicators to 1h timeframe
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Get 1d data for EMA50 regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for stoploss calculation (1h ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA20 (20), EMA50 (50), ATR (14)
    start_idx = max(20, 50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(r1_4h_aligned[i]) or
            np.isnan(s1_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        ema_20_val = ema_20_4h_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        r1_val = r1_4h_aligned[i]
        s1_val = s1_4h_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R1, above 4h EMA20, and above 1d EMA50 (bullish regime)
            long_signal = (close_val > r1_val) and (close_val > ema_20_val) and (close_val > ema_50_val)
            
            # Short: price breaks below S1, below 4h EMA20, and below 1d EMA50 (bearish regime)
            short_signal = (close_val < s1_val) and (close_val < ema_20_val) and (close_val < ema_50_val)
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price breaks below S1 OR ATR stoploss (2*ATR below entry) OR regime change (below 1d EMA50)
            if (close_val < s1_val) or (close_val < entry_price - 2.0 * atr_val) or (close_val < ema_50_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price breaks above R1 OR ATR stoploss (2*ATR above entry) OR regime change (above 1d EMA50)
            if (close_val > r1_val) or (close_val > entry_price + 2.0 * atr_val) or (close_val > ema_50_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dEMAFilter_v1"
timeframe = "1h"
leverage = 1.0