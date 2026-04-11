#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Strategy: 1d Camarilla pivot levels with volume confirmation and 1w trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels identify key support/resistance where price often reverses or breaks.
# In trending markets (1w EMA50), breakouts above/below pivot levels with volume continuation yield high-probability trades.
# Low frequency (~10-20/year) to minimize fee drag and improve generalization.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    # Camarilla formulas: 
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # We use H3/L3 for breakout entries, H4/L4 for stop levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Set first day's values to avoid NaN
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    daily_range = prev_high - prev_low
    H3 = prev_close + 1.0 * daily_range
    L3 = prev_close - 1.0 * daily_range
    H4 = prev_close + 1.5 * daily_range
    L4 = prev_close - 1.5 * daily_range
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 since we use previous day's data
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(H3[i]) or np.isnan(L3[i]) or
            np.isnan(H4[i]) or np.isnan(L4[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry logic: Breakout of H3/L3 with volume confirmation and trend alignment
        if (close[i] > H3[i] and vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < L3[i] and vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Price reaches H4/L4 (stop levels) or trend reversal
        elif position == 1 and (close[i] >= H4[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= L4[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals