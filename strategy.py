#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_v2
# Strategy: 4h Camarilla pivot breakout with volume confirmation and daily trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels (from daily high/low/close) act as strong support/resistance.
# Breakouts above H4 (resistance) or below L4 (support) with volume confirmation and
# alignment with daily trend (price vs daily EMA50) capture institutional moves.
# Works in bull markets (breakouts above H4 in uptrend) and bear markets (breakdowns below L4 in downtrend).
# Volume confirms breakout sincerity. Target: 25-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from daily OHLC
    # Typical Camarilla: based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate levels: H4, L4 (primary breakout levels)
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * range_1d
    camarilla_l4 = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + trend alignment
        if close[i] > camarilla_h4_aligned[i] and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] < camarilla_l4_aligned[i] and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to previous day's close (pivot point)
        elif position == 1 and close[i] < camarilla_h4_aligned[i] - 0.5 * range_1d[i]:  # midpoint toward L4
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > camarilla_l4_aligned[i] + 0.5 * range_1d[i]:  # midpoint toward H4
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals