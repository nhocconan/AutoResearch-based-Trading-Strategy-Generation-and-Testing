#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Camarilla pivot with daily volume and weekly trend filter
# Hypothesis: Camarilla levels from daily charts act as strong support/resistance in 12h timeframe.
# Volume confirms institutional interest at these levels. Weekly trend filter ensures we trade
# with the higher timeframe momentum. Works in bull (buying dips in uptrend) and bear (selling rallies in downtrend).
# Target: 15-30 trades/year to minimize fee drag.
name = "12h_camarilla_pivot_1d_volume_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from daily data
    # Based on previous day's OHLC
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3.0
    range_val = daily_high - daily_low
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = Close + Range * 1.1/2
    # H3 = Close + Range * 1.1/4
    # H2 = Close + Range * 1.1/6
    # H1 = Close + Range * 1.1/12
    # L1 = Close - Range * 1.1/12
    # L2 = Close - Range * 1.1/6
    # L3 = Close - Range * 1.1/4
    # L4 = Close - Range * 1.1/2
    
    h4 = daily_close + range_val * 1.1 / 2.0
    h3 = daily_close + range_val * 1.1 / 4.0
    h2 = daily_close + range_val * 1.1 / 6.0
    h1 = daily_close + range_val * 1.1 / 12.0
    l1 = daily_close - range_val * 1.1 / 12.0
    l2 = daily_close - range_val * 1.1 / 6.0
    l3 = daily_close - range_val * 1.1 / 4.0
    l4 = daily_close - range_val * 1.1 / 2.0
    
    # Calculate weekly trend using 21-period EMA
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, min_periods=21, adjust=False).mean().values
    weekly_trend = weekly_close > weekly_ema  # True for uptrend
    
    # Calculate daily 20-period volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF data to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(weekly_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below L3 (strong support broken)
            if close[i] < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above H3 (strong resistance broken)
            if close[i] > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price touches L4 with volume confirmation and weekly uptrend
            if (close[i] <= l4_aligned[i] * 1.002 and  # Allow small buffer for touch
                vol_confirm and 
                weekly_trend_aligned[i] > 0.5):  # Weekly uptrend
                position = 1
                signals[i] = 0.25
            # Enter short: price touches H4 with volume confirmation and weekly downtrend
            elif (close[i] >= h4_aligned[i] * 0.998 and  # Allow small buffer for touch
                  vol_confirm and 
                  weekly_trend_aligned[i] < 0.5):  # Weekly downtrend
                position = -1
                signals[i] = -0.25
    
    return signals