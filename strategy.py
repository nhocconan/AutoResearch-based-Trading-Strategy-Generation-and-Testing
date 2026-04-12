#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h_1d_camarilla_breakout_v1
# Uses daily and 4h Camarilla pivot levels to identify key support/resistance.
# Long when price breaks above daily H4 with volume confirmation and 4h trend up.
# Short when price breaks below daily L4 with volume confirmation and 4h trend down.
# Exits when price returns to daily pivot point (mean reversion).
# Uses 4h for trend direction, 1h for entry timing to reduce false breakouts.
# Target: 15-35 trades/year (60-140 over 4 years) to minimize fee drag.
# Works in trending markets via breakouts and ranging via mean reversion to pivot.

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation (key levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels for each day
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H4 = PP + 1.1/2 * range, L4 = PP - 1.1/2 * range
    h4_1d = pp_1d + (1.1 / 2) * range_1d
    l4_1d = pp_1d - (1.1 / 2) * range_1d
    
    # Align daily levels to 1h timeframe (daily values update after daily bar closes)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 1.5 * 20-period average (1h timeframe)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (reduce noise outside active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(ema_4h_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above daily H4 with 4h uptrend
        if close[i] > h4_1d_aligned[i] and ema_4h_aligned[i] > close[i] and position != 1:
            position = 1
            signals[i] = 0.20
        # Short signal: price breaks below daily L4 with 4h downtrend
        elif close[i] < l4_1d_aligned[i] and ema_4h_aligned[i] < close[i] and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit conditions: price returns to daily pivot point (mean reversion)
        elif position == 1 and close[i] <= pp_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= pp_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals