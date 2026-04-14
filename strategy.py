#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        multiplier = 2 / (20 + 1)
        ema_20_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(df_1d)):
            ema_20_1d[i] = (close_1d[i] - ema_20_1d[i-1]) * multiplier + ema_20_1d[i-1]
    
    # Align 1d EMA20 to daily timeframe (no additional alignment needed as we're on 1d)
    ema_20_1d_aligned = ema_20_1d  # Already on 1d timeframe
    
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
    
    # Align daily ATR to daily timeframe
    atr_1d_aligned = atr_1d  # Already on 1d timeframe
    
    # Calculate daily volume moving average (20-period)
    volume_ma = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        for i in range(19, len(df_1d)):
            volume_ma[i] = np.mean(df_1d['volume'].values[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, len(df_1d)):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_1d_aligned[i] / close_1d[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 60% of 20-period MA)
        if df_1d['volume'].values[i] < 0.6 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Calculate pivot levels based on previous day's range
        if i >= 1:
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            prev_range = prev_high - prev_low
            
            # S3 and R3 levels (extreme rejection zones)
            s3 = prev_close - (prev_range * 1.1)
            r3 = prev_close + (prev_range * 1.1)
            
            if position == 0:
                # Long: Price rejects S3 with volume and trend alignment
                if low_1d[i] <= s3 and close_1d[i] > s3 and df_1d['volume'].values[i] > volume_ma[i] and close_1d[i] > ema_20_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: Price rejects R3 with volume and trend alignment
                elif high_1d[i] >= r3 and close_1d[i] < r3 and df_1d['volume'].values[i] > volume_ma[i] and close_1d[i] < ema_20_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit: Price breaks S3 again or reaches mean reversion target
                # Calculate S1 for profit target (mean reversion level)
                s1 = prev_close - (prev_range * 0.5)
                if low_1d[i] <= s3 or close_1d[i] <= s1:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price breaks R3 again or reaches mean reversion target
                # Calculate R1 for profit target (mean reversion level)
                r1 = prev_close + (prev_range * 0.5)
                if high_1d[i] >= r3 or close_1d[i] >= r1:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Pivot_S3R3_Rejection_Volume_Filter_v3"
timeframe = "1d"
leverage = 1.0