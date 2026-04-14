# 4h_1d_Pivot_S3R3_Rejection_Volume_Filter
# Hypothesis: Uses daily pivot S3/R3 levels for mean reversion entries in both bull and bear markets.
# Price rejecting extreme S3/R3 levels with volume confirmation indicates potential reversal.
# Works in bull markets (buying dips at S3) and bear markets (selling rallies at R3).
# Volume filter ensures institutional participation, reducing false signals.
# Timeframe: 4h balances trade frequency and signal quality.

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
    
    # Calculate 1d EMA21 for trend filter
    close_1d = df_1d['close'].values
    ema_21_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 21:
        multiplier = 2 / (21 + 1)
        ema_21_1d[20] = np.mean(close_1d[:21])
        for i in range(21, len(df_1d)):
            ema_21_1d[i] = (close_1d[i] - ema_21_1d[i-1]) * multiplier + ema_21_1d[i-1]
    
    # Align 1d EMA21 to 4h timeframe
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 1d ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # Calculate 4h volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_1d_aligned[i]) or
            np.isnan(atr_4h_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_4h_aligned[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 60% of 20-period MA)
        if volume[i] < 0.6 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Calculate pivot levels based on previous day's range
        # Need previous day's data - use index-1 for daily data alignment
        if i >= 1:
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            prev_range = prev_high - prev_low
            
            # S3 and R3 levels (extreme rejection zones)
            s3 = prev_close - (prev_range * 1.1)
            r3 = prev_close + (prev_range * 1.1)
            
            # Align S3/R3 to 4h timeframe (constant values for the day)
            s3_4h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s3))[i]
            r3_4h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r3))[i]
            
            if position == 0:
                # Long: Price rejects S3 with volume and trend alignment
                if low[i] <= s3 and close[i] > s3 and volume[i] > volume_ma[i] and close[i] > ema_21_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: Price rejects R3 with volume and trend alignment
                elif high[i] >= r3 and close[i] < r3 and volume[i] > volume_ma[i] and close[i] < ema_21_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit: Price breaks S3 again or reaches mean reversion target
                # Calculate S1 for profit target (mean reversion level)
                s1 = prev_close - (prev_range * 0.5)
                s1_4h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s1))[i]
                if low[i] <= s3 or close[i] <= s1_4h:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price breaks R3 again or reaches mean reversion target
                # Calculate R1 for profit target (mean reversion level)
                r1 = prev_close + (prev_range * 0.5)
                r1_4h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r1))[i]
                if high[i] >= r3 or close[i] >= r1_4h:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Pivot_S3R3_Rejection_Volume_Filter"
timeframe = "4h"
leverage = 1.0