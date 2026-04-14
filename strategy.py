#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day pivot rejection + volume confirmation + EMA200 trend filter
# Works in bull/bear: Pivot rejection captures mean-reversion at key levels, volume confirms institutional interest, EMA200 ensures trend alignment
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 200:
        ema_series = pd.Series(close_1d)
        ema200_1d = ema_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMA200 to 12h timeframe
    ema200_12h_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 12h volume moving average (20-period) for volume filter
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        volume_series = pd.Series(volume)
        volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(ema200_12h_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR proxy: avoid tight ranges)
        # Use 12h price range as volatility filter
        if i >= 1:
            price_range = (high[i] - low[i]) / close[i]
            if price_range < 0.005:  # Skip if daily range < 0.5%
                signals[i] = 0.0
                continue
        
        # Skip low volume periods (volume < 70% of 20-period MA)
        if volume[i] < 0.7 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Get previous day's data (1d index)
        if i >= 1:
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            
            # Calculate pivot points (standard formula)
            pivot = (prev_high + prev_low + prev_close) / 3.0
            s1 = (2 * pivot) - prev_high
            r1 = (2 * pivot) - prev_low
            s2 = pivot - (prev_high - prev_low)
            r2 = pivot + (prev_high - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)
            r3 = prev_high + 2 * (pivot - prev_low)
            
            # Align S3/R3 to 12h timeframe (constant values for the day)
            s3_array = np.full(len(df_1d), s3)
            r3_array = np.full(len(df_1d), r3)
            s3_12h = align_htf_to_ltf(prices, df_1d, s3_array)[i]
            r3_12h = align_htf_to_ltf(prices, df_1d, r3_array)[i]
            
            if position == 0:
                # Long: Price rejects S3 with volume and above EMA200 (bullish trend)
                if low[i] <= s3_12h and close[i] > s3_12h and volume[i] > volume_ma[i] and close[i] > ema200_12h_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: Price rejects R3 with volume and below EMA200 (bearish trend)
                elif high[i] >= r3_12h and close[i] < r3_12h and volume[i] > volume_ma[i] and close[i] < ema200_12h_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit: Price breaks S3 again or trend changes (price below EMA200)
                if low[i] <= s3_12h or close[i] < ema200_12h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price breaks R3 again or trend changes (price above EMA200)
                if high[i] >= r3_12h or close[i] > ema200_12h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1d_Pivot_S3R3_Rejection_Volume_EMA200_Filter_v3"
timeframe = "12h"
leverage = 1.0