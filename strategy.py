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
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly ATR for volatility regime filter
    high_low = high_1w - low_1w
    high_close = np.abs(high_1w - np.roll(close_1w, 1))
    low_close = np.abs(low_1w - np.roll(close_1w, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate 6-hour Donchian channels (20-period) for breakout signals
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    for i in range(20, n):
        donchian_high[i] = high_series.iloc[i-20:i].max()
        donchian_low[i] = low_series.iloc[i-20:i].min()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(atr_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            continue
        
        # Get previous week's data (1w index)
        if i >= 1:
            prev_week_close = close_1w[i-1]
            prev_week_high = high_1w[i-1]
            prev_week_low = low_1w[i-1]
            
            # Calculate weekly pivot points (standard formula)
            pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
            s1 = (2 * pivot) - prev_week_high
            r1 = (2 * pivot) - prev_week_low
            s2 = pivot - (prev_week_high - prev_week_low)
            r2 = pivot + (prev_week_high - prev_week_low)
            s3 = prev_week_low - 2 * (prev_week_high - pivot)
            r3 = prev_week_high + 2 * (pivot - prev_week_low)
            
            # Align S3/R3 to weekly timeframe (constant values for the week)
            s3_array = np.full(len(df_1w), s3)
            r3_array = np.full(len(df_1w), r3)
            s3_1w = align_htf_to_ltf(prices, df_1w, s3_array)[i]
            r3_1w = align_htf_to_ltf(prices, df_1w, r3_array)[i]
            
            # Calculate volatility filter based on weekly ATR
            vol_threshold = np.percentile(atr_1w_aligned[max(0, i-50):i+1], 40) if i >= 50 else np.mean(atr_1w_aligned[max(0, i-10):i+1])
            
            if position == 0:
                # Long: Price breaks above R3 with volume and volatility filter
                vol_ma = np.mean(volume[i-5:i]) if i >= 5 else volume[i]
                if (close[i] > r3_1w and close[i-1] <= r3_1w and 
                    volume[i] > vol_ma * 1.5 and 
                    close[i] > donchian_high[i-1] and  # Additional breakout confirmation
                    atr_1w_aligned[i] > vol_threshold):  # Volatility filter - only trade in higher vol regimes
                    position = 1
                    signals[i] = position_size
                # Short: Price breaks below S3 with volume and volatility filter
                elif (close[i] < s3_1w and close[i-1] >= s3_1w and 
                      volume[i] > vol_ma * 1.5 and 
                      close[i] < donchian_low[i-1] and  # Additional breakdown confirmation
                      atr_1w_aligned[i] > vol_threshold):  # Volatility filter - only trade in higher vol regimes
                    position = -1
                    signals[i] = -position_size
            elif position == 1:
                # Exit: Price breaks below S3 or volatility drops significantly
                if close[i] < s3_1w or atr_1w_aligned[i] < vol_threshold * 0.7:
                    position = 0
                    signals[i] = 0.0
            elif position == -1:
                # Exit: Price breaks above R3 or volatility drops significantly
                if close[i] > r3_1w or atr_1w_aligned[i] < vol_threshold * 0.7:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_S3R3_Breakout_VolFilter"
timeframe = "6h"
leverage = 1.0