#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Pivot Points with price action confirmation
# - Long when price closes above weekly pivot with volume confirmation in bullish trend
# - Short when price closes below weekly pivot with volume confirmation in bearish trend
# - Uses weekly high/low/close to calculate pivot and support/resistance levels
# - Adds 1w EMA50 trend filter to align with higher timeframe trend
# - Designed to work in both trending and ranging markets with clear entry/exit rules
# - Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_WeeklyPivot_PB_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    prev_high = df_1w['high'].shift(1)
    prev_low = df_1w['low'].shift(1)
    prev_close = df_1w['close'].shift(1)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Align weekly pivot levels to daily timeframe
    pivot_1d = align_htf_to_ltf(prices, df_1w, pivot.values)
    r1_1d = align_htf_to_ltf(prices, df_1w, r1.values)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1.values)
    r2_1d = align_htf_to_ltf(prices, df_1w, r2.values)
    s2_1d = align_htf_to_ltf(prices, df_1w, s2.values)
    
    # 1w EMA50 for trend filter
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.2 * vol_ma_20)
    
    # Avoid trading in extremely low volume conditions
    vol_filter = volume > (vol_ma_20 * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or insufficient volume
        if (np.isnan(pivot_1d[i]) or np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or
            np.isnan(r2_1d[i]) or np.isnan(s2_1d[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price closes above pivot with volume in bullish trend
            if (close[i] > pivot_1d[i] and 
                close[i] > close[i-1] and  # Price momentum confirmation
                volume_filter[i] and 
                vol_filter[i] and
                close[i] > ema_50_1w_aligned[i]):  # Only in uptrend
                signals[i] = 0.25
                position = 1
            # Short entry: price closes below pivot with volume in bearish trend
            elif (close[i] < pivot_1d[i] and 
                  close[i] < close[i-1] and  # Price momentum confirmation
                  volume_filter[i] and 
                  vol_filter[i] and
                  close[i] < ema_50_1w_aligned[i]):  # Only in downtrend
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below pivot or hits S1/S2
            if close[i] < pivot_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above pivot or hits R1/R2
            if close[i] > pivot_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals