#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1S1_Breakout_Trend_Filter_v1
Hypothesis: Weekly pivot levels (R1/S1) act as strong support/resistance on daily timeframe.
Buy when price breaks above R1 with bullish trend filter (price > 50 EMA), sell when breaks below S1 with bearish trend (price < 50 EMA).
Uses volume confirmation to avoid false breakouts. Designed for low frequency (~10-15 trades/year) to minimize fee drag.
Works in bull via trend-following breaks and bear via mean-reversion at extreme pivots.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot points from previous week
    df_1w = get_htf_data(prices, '1w')
    # Need at least 2 weeks of data to calculate pivots from previous week
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Use previous week's OHLC to calculate pivot points for current week
    # Shift by 1 to use previous week's data
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Calculate pivot points
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    
    # Align to daily timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # EMA trend filter (50-period on daily)
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need enough data for EMA
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with bullish trend and volume
            if price > r1_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with bearish trend and volume
            elif price < s1_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below pivot or trend turns bearish
            if price < pivot_aligned[i] or price < ema_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above pivot or trend turns bullish
            if price > pivot_aligned[i] or price > ema_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Pivot_R1S1_Breakout_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0