#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot level touch with volume spike and weekly trend filter.
# Uses weekly close > 40-week EMA for trend filter, long when price touches S1/S2 with volume spike,
# short when price touches R1/R2 with volume spike. Exits when price moves to opposite H4/L4 level.
# Designed for 15-25 trades/year with high win rate via confluence of support/resistance,
# volume confirmation, and trend alignment. Works in bull via longs at support, in bear via shorts at resistance.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 40-week EMA for trend filter
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # H2 = close + 1.166 * (high - low)
    # H1 = close + 1.083 * (high - low)
    # L1 = close - 1.083 * (high - low)
    # L2 = close - 1.166 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # L4 = close - 1.5 * (high - low)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day has no previous
    prev_high[0] = prev_low[0] = prev_close[0] = close_1d[0]
    
    diff = prev_high - prev_low
    
    H4 = prev_close + 1.5 * diff
    H3 = prev_close + 1.25 * diff
    H2 = prev_close + 1.166 * diff
    H1 = prev_close + 1.083 * diff
    L1 = prev_close - 1.083 * diff
    L2 = prev_close - 1.166 * diff
    L3 = prev_close - 1.25 * diff
    L4 = prev_close - 1.5 * diff
    
    # Align Camarilla levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume filter: volume > 2.0 x 24-period average (24*12h = 12 days)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 24-period volume MA and first valid pivot levels
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(ema40_1w_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filter from weekly EMA40
        bullish_trend = ema40_1w_aligned[i] > 0 and close[i] > ema40_1w_aligned[i]
        bearish_trend = ema40_1w_aligned[i] > 0 and close[i] < ema40_1w_aligned[i]
        
        if position == 0:
            # Long: price touches L1 or L2 with volume spike and bullish weekly trend
            if bullish_trend and vol_filter and (
                abs(price - L1_aligned[i]) < 0.002 * price or  # within 0.2% of L1
                abs(price - L2_aligned[i]) < 0.002 * price):   # within 0.2% of L2
                signals[i] = size
                position = 1
            # Short: price touches H1 or H2 with volume spike and bearish weekly trend
            elif bearish_trend and vol_filter and (
                abs(price - H1_aligned[i]) < 0.002 * price or  # within 0.2% of H1
                abs(price - H2_aligned[i]) < 0.002 * price):   # within 0.2% of H2
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H4 (opposite resistance) or volume drops
            if price >= H4_aligned[i] * 0.998 or vol_now <= vol_avg:  # within 0.2% of H4 or low volume
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches L4 (opposite support) or volume drops
            if price <= L4_aligned[i] * 1.002 or vol_now <= vol_avg:  # within 0.2% of L4 or low volume
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_Pivot_Touch_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0