# 12h_1w_1d_S3R3_Pivot_Breakout_With_Volume_Confirmation
# Hypothesis: Price breaking daily S3/R3 levels with volume confirmation on 12h timeframe
# captures institutional order flow during extreme price rejection. Works in both bull/bear
# markets because S3/R3 act as dynamic support/resistance based on previous day's range.
# Volume filter ensures only significant breakouts trigger trades, reducing false signals.
# 12h timeframe balances trade frequency (target 15-30/year) with sufficient signal clarity.
# Weekly trend filter (EMA50) avoids counter-trend trades in strong trends.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Load daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(df_1w)):
            ema_50_1w[i] = (close_1w[i] - ema_50_1w[i-1]) * multiplier + ema_50_1w[i-1]
    
    # Align weekly EMA to 12h timeframe
    ema_50_12h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily ATR (14-period) for volatility filter
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
    
    # Align daily ATR to 12h timeframe
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12-hour volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_12h[i]) or
            np.isnan(atr_12h[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_12h[i] / close[i] < 0.003:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 50% of 20-period MA)
        if volume[i] < 0.5 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Calculate pivot levels based on previous day's range
        # Need previous day's data - use index-1 for daily data alignment
        if i >= 1:
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            prev_range = prev_high - prev_low
            
            # S3 and R3 levels (more extreme than standard S1/R1)
            s3 = prev_close - (prev_range * 1.1)
            r3 = prev_close + (prev_range * 1.1)
            
            # Align S3/R3 to 12h timeframe (constant values for the day)
            s3_12h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s3))[i]
            r3_12h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r3))[i]
            
            if position == 0:
                # Long: Price breaks below S3 (extreme rejection) AND closes back above S3
                # with volume confirmation AND above weekly EMA50 (trend alignment)
                if low[i] <= s3 and close[i] > s3 and volume[i] > volume_ma[i] and close[i] > ema_50_12h[i]:
                    position = 1
                    signals[i] = position_size
                # Short: Price breaks above R3 (extreme rejection) AND closes back below R3
                # with volume confirmation AND below weekly EMA50 (trend alignment)
                elif high[i] >= r3 and close[i] < r3 and volume[i] > volume_ma[i] and close[i] < ema_50_12h[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit: Price breaks below S3 again or reaches R1 (mean reversion target)
                # Calculate R1 for profit target
                r1 = prev_close + (prev_range * 0.5)
                r1_12h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r1))[i]
                if low[i] <= s3 or close[i] >= r1_12h:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price breaks above R3 again or reaches S1 (mean reversion target)
                # Calculate S1 for profit target
                s1 = prev_close - (prev_range * 0.5)
                s1_12h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s1))[i]
                if high[i] >= r3 or close[i] <= s1_12h:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1w_1d_S3R3_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0