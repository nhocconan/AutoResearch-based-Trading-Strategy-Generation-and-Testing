#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h_1d_camarilla_breakout_v1
# Uses 4h and 1d timeframes for signal direction, 1h for entry timing.
# Long when price breaks above 4h H4 with 1d trend filter (price > 1d EMA200) and volume confirmation.
# Short when price breaks below 4h L4 with 1d trend filter (price < 1d EMA200) and volume confirmation.
# Exits when price returns to 4h pivot point (PP) or trend filter fails.
# Designed for low trade frequency (target: 15-37 trades/year) to minimize fee drag.
# Works in trending markets via breakouts and uses trend filter to avoid false signals in ranging markets.

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels based on previous 4h bar's OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivot point and Camarilla levels for each 4h bar
    pp_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    
    # Camarilla levels: H4 = PP + 1.1/2 * range, L4 = PP - 1.1/2 * range
    h4_4h = pp_4h + (1.1 / 2) * range_4h
    l4_4h = pp_4h - (1.1 / 2) * range_4h
    
    # Align 4h levels to 1h timeframe (4h values update after 4h bar closes)
    h4_4h_aligned = align_htf_to_ltf(prices, df_4h, h4_4h)
    l4_4h_aligned = align_htf_to_ltf(prices, df_4h, l4_4h)
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average (1h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(h4_4h_aligned[i]) or np.isnan(l4_4h_aligned[i]) or 
            np.isnan(pp_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
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
        
        # Long signal: price breaks above 4h H4 with 1d uptrend filter
        if (close[i] > h4_4h_aligned[i] and 
            close[i] > ema_200_1d_aligned[i] and 
            position != 1):
            position = 1
            signals[i] = 0.20
        # Short signal: price breaks below 4h L4 with 1d downtrend filter
        elif (close[i] < l4_4h_aligned[i] and 
              close[i] < ema_200_1d_aligned[i] and 
              position != -1):
            position = -1
            signals[i] = -0.20
        # Exit conditions: price returns to 4h pivot point or trend filter fails
        elif position == 1 and (close[i] <= pp_4h_aligned[i] or close[i] <= ema_200_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= pp_4h_aligned[i] or close[i] >= ema_200_1d_aligned[i]):
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