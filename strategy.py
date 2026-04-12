#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_1w_camarilla_breakout_v2
# Uses daily and weekly CAMARILLA PIVOT LEVELS (H4/L4) as key S/R on 12h chart.
# LONG when price breaks above DAILY H4 with volume confirmation (>1.5x 20-period avg).
# SHORT when price breaks below DAILY L4 with volume confirmation.
# EXIT when price returns to DAILY PIVOT POINT (PP).
# WEEKLY CONFIRMATION: Only take long if WEEKLY trend is UP (price > WEEKLY VWAP).
# Only take short if WEEKLY trend is DOWN (price < WEEKLY VWAP).
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
# Works in trending markets via breakouts and in ranging via mean reversion to daily pivot.
# Weekly trend filter reduces whipsaw in sideways markets.

name = "12h_1d_1w_camarilla_breakout_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate DAILY Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels for each day
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H4 = PP + 1.1/2 * range, L4 = PP - 1.1/2 * range
    h4_1d = pp_1d + (1.1 / 2) * range_1d
    l4_1d = pp_1d - (1.1 / 2) * range_1d
    
    # Align daily levels to 12h timeframe (daily values update after daily bar closes)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Calculate WEEKLY VWAP for trend filter
    # VWAP = sum(price * volume) / sum(volume) for the week
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    vwap_num = (typical_price_1w * df_1w['volume']).cumsum()
    vwap_den = df_1w['volume'].cumsum()
    vwap_1w = (vwap_num / vwap_den).values
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period average (12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(vwap_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter: only long if price > weekly VWAP, short if price < weekly VWAP
        weekly_uptrend = close[i] > vwap_1w_aligned[i]
        weekly_downtrend = close[i] < vwap_1w_aligned[i]
        
        # LONG signal: price breaks above daily H4 AND weekly uptrend
        if close[i] > h4_1d_aligned[i] and weekly_uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        # SHORT signal: price breaks below daily L4 AND weekly downtrend
        elif close[i] < l4_1d_aligned[i] and weekly_downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # EXIT conditions: price returns to daily pivot point (mean reversion)
        elif position == 1 and close[i] <= pp_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= pp_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals