# 12h_1d_Camarilla_100_Period_Volume_RSI_Trend
# Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance levels. 
# Combined with 100-period EMA trend filter and volume confirmation, this creates high-probability entries.
# The 100-period EMA ensures we only trade with the higher timeframe trend, reducing whipsaws.
# Volume > 1.5x average confirms institutional interest. RSI 50 threshold ensures momentum alignment.
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).
# Target: 15-25 trades/year on 12h timeframe to minimize fee drag.

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
    
    # Load daily data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (daily)
    # Formula: Pivot = (H + L + C) / 3
    # Range = H - L
    # Resistance levels: R1 = C + (Range * 1.1/12), R2 = C + (Range * 1.1/6), R3 = C + (Range * 1.1/4), R4 = C + (Range * 1.1/2)
    # Support levels: S1 = C - (Range * 1.1/12), S2 = C - (Range * 1.1/6), S3 = C - (Range * 1.1/4), S4 = C - (Range * 1.1/2)
    # We'll use S3 and R3 as primary levels (more significant than S1/R1)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # S3 and R3 levels
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Calculate 100-period EMA for trend filter (daily)
    ema100_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 100:
        ema100_1d[99] = np.mean(close_1d[:100])
        for i in range(100, len(close_1d)):
            alpha = 2.0 / (100 + 1)  # 0.0198
            ema100_1d[i] = close_1d[i] * alpha + ema100_1d[i-1] * (1 - alpha)
    
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Calculate 14-period RSI for momentum (daily)
    rsi14_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 14:
        delta = np.diff(close_1d, prepend=close_1d[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_1d, np.nan)
        avg_loss = np.full_like(close_1d, np.nan)
        
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        rs = np.full_like(close_1d, np.nan)
        for i in range(13, len(close_1d)):
            if avg_loss[i] > 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi14_1d[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi14_1d[i] = 100 if avg_gain[i] > 0 else 0
    
    rsi14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi14_1d)
    
    # Volume ratio: current 12h volume vs 20-period average
    vol_ma_20 = np.full_like(volume, np.nan)
    for j in range(19, len(volume)):
        vol_ma_20[j] = np.mean(volume[j-19:j+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or 
            np.isnan(ema100_1d_aligned[i]) or 
            np.isnan(rsi14_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: Price above S3 + price above EMA100 + RSI > 50 + volume confirmation
            if (close[i] > s3_1d_aligned[i] and
                close[i] > ema100_1d_aligned[i] and
                rsi14_1d_aligned[i] > 50 and
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: Price below R3 + price below EMA100 + RSI < 50 + volume confirmation
            elif (close[i] < r3_1d_aligned[i] and
                  close[i] < ema100_1d_aligned[i] and
                  rsi14_1d_aligned[i] < 50 and
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price below S3 OR RSI < 40
            if (close[i] < s3_1d_aligned[i] or 
                rsi14_1d_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price above R3 OR RSI > 60
            if (close[i] > r3_1d_aligned[i] or 
                rsi14_1d_aligned[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_100_Period_Volume_RSI_Trend"
timeframe = "12h"
leverage = 1.0