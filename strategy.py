#!/usr/bin/env python3
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Close for pivot calculations
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Pivot Point and support/resistance levels
    # Pivot = (High + Low + Close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*Pivot - Low
    r1_1d = 2 * pivot_1d - low_1d
    # S1 = 2*Pivot - High
    s1_1d = 2 * pivot_1d - high_1d
    # R2 = Pivot + (High - Low)
    r2_1d = pivot_1d + (high_1d - low_1d)
    # S2 = Pivot - (High - Low)
    s2_1d = pivot_1d - (high_1d - low_1d)
    # R3 = High + 2*(Pivot - Low)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    # S3 = Low - 2*(High - Pivot)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 60-period moving average for trend filter (6h timeframe)
    ma_60 = pd.Series(close).rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or
            np.isnan(s2_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(ma_60[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.005  # Minimum 0.5% ATR relative to price
        
        # Trend filter: price above/below 60-period MA
        trend_filter_long = price > ma_60[i]
        trend_filter_short = price < ma_60[i]
        
        if position == 0:
            # Long setup: price breaks above R2 with volume confirmation
            long_breakout = price > r2_1d_aligned[i]
            # Short setup: price breaks below S2 with volume confirmation
            short_breakout = price < s2_1d_aligned[i]
            
            if long_breakout and vol_filter and trend_filter_long:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_filter and trend_filter_short:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below pivot or S1 (stop loss)
            if price < pivot_1d_aligned[i] or price < s1_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above pivot or R1 (stop loss)
            if price > pivot_1d_aligned[i] or price > r1_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dPivot_R2S2_Breakout_v1"
timeframe = "6h"
leverage = 1.0