#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA20 for trend filter
    ema_20_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        ema_series = pd.Series(close_1d).ewm(span=20, adjust=False).mean()
        ema_20_1d = ema_series.values
    
    # Align 1d EMA20 to 4h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 1d ATR (14-period) for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily ATR to 4h timeframe
    atr_4h_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h volume moving average (20-period) for volume filter
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        volume_series = pd.Series(volume)
        volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Conservative size to manage drawdown
    
    for i in range(60, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1d_aligned[i]) or
            np.isnan(atr_4h_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_4h_aligned[i] / close[i] < 0.003:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 60% of 20-period MA)
        if volume[i] < 0.6 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Calculate pivot levels based on previous day's close
        if i >= 1:
            # Get previous day's data (1d index)
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
            
            # Align S3/R3 to 4h timeframe (constant values for the day)
            s3_array = np.full(len(df_1d), s3)
            r3_array = np.full(len(df_1d), r3)
            s3_4h = align_htf_to_ltf(prices, df_1d, s3_array)[i]
            r3_4h = align_htf_to_ltf(prices, df_1d, r3_array)[i]
            
            if position == 0:
                # Long: Price rejects S3 with volume and trend alignment
                if low[i] <= s3_4h and close[i] > s3_4h and volume[i] > volume_ma[i] and close[i] > ema_20_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: Price rejects R3 with volume and trend alignment
                elif high[i] >= r3_4h and close[i] < r3_4h and volume[i] > volume_ma[i] and close[i] < ema_20_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit: Price breaks S3 again or reaches mean reversion target
                # Dynamic exit at 50% of the range from S3 to pivot
                exit_level = s3_4h + (pivot - s3_4h) * 0.5
                if low[i] <= s3_4h or close[i] <= exit_level:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price breaks R3 again or reaches mean reversion target
                # Dynamic exit at 50% of the range from pivot to R3
                exit_level = r3_4h - (r3_4h - pivot) * 0.5
                if high[i] >= r3_4h or close[i] >= exit_level:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Pivot_S3R3_Rejection_Volume_Filter_v5"
timeframe = "4h"
leverage = 1.0