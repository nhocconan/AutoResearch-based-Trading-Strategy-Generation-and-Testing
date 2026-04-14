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
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d close for pivot calculations
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter (more stable than SMA)
    ema_50_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 50:
        # Use pandas EMA for accuracy and efficiency
        ema_series = pd.Series(close_1d).ewm(span=50, adjust=False).mean()
        ema_50_1d = ema_series.values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # ATR with proper smoothing
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
        # Use pandas rolling mean for efficiency
        volume_series = pd.Series(volume)
        volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Reduced size for better risk management
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or
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
            
            # Calculate dynamic pivot levels based on volatility
            # Using ATR-based levels instead of fixed percentages
            atr_value = atr_1d[i-1]
            pivot_range = atr_value * 2.5  # 2.5x ATR for pivot width
            
            # S3 and R3 levels (extreme rejection zones)
            s3 = prev_close - pivot_range
            r3 = prev_close + pivot_range
            
            # Align S3/R3 to 4h timeframe (constant values for the day)
            s3_array = np.full(len(df_1d), s3)
            r3_array = np.full(len(df_1d), r3)
            s3_4h = align_htf_to_ltf(prices, df_1d, s3_array)[i]
            r3_4h = align_htf_to_ltf(prices, df_1d, r3_array)[i]
            
            if position == 0:
                # Long: Price rejects S3 with volume and trend alignment
                if low[i] <= s3 and close[i] > s3 and volume[i] > volume_ma[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: Price rejects R3 with volume and trend alignment
                elif high[i] >= r3 and close[i] < r3 and volume[i] > volume_ma[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit: Price breaks S3 again or reaches mean reversion target
                # Calculate dynamic exit level
                exit_level = prev_close - (pivot_range * 0.3)  # 30% of pivot range
                exit_array = np.full(len(df_1d), exit_level)
                exit_4h = align_htf_to_ltf(prices, df_1d, exit_array)[i]
                if low[i] <= s3 or close[i] <= exit_4h:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price breaks R3 again or reaches mean reversion target
                # Calculate dynamic exit level
                exit_level = prev_close + (pivot_range * 0.3)  # 30% of pivot range
                exit_array = np.full(len(df_1d), exit_level)
                exit_4h = align_htf_to_ltf(prices, df_1d, exit_array)[i]
                if high[i] >= r3 or close[i] >= exit_4h:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Pivot_S3R3_Rejection_Volume_Filter_v4"
timeframe = "4h"
leverage = 1.0