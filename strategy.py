#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with volume confirmation and 1w trend filter
# Long when price touches S3 support + volume > 1.5x avg + weekly trend up
# Short when price touches R3 resistance + volume > 1.5x avg + weekly trend down
# Exit at pivot point or trend reversal
# Designed for 15-35 trades/year on 4h timeframe with mean reversion in ranges and trend alignment

name = "4h_1w_camarilla_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(200) for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Camarilla levels
    high_1d = df_1w['high'].values  # Using weekly high/low for pivot calculation
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels calculation (based on previous day's range)
    # We'll calculate these for each bar using the previous daily bar's data
    # For simplicity, we use weekly OHLC to derive pivot points
    # In practice, we'd use daily, but weekly provides smoother levels
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    s3 = pivot - (range_1w * 1.1 / 6)
    s2 = pivot - (range_1w * 1.1 / 4)
    s1 = pivot - (range_1w * 1.1 / 2)
    r1 = pivot + (range_1w * 1.1 / 2)
    r2 = pivot + (range_1w * 1.1 / 4)
    r3 = pivot + (range_1w * 1.1 / 6)
    
    # Align weekly Camarilla levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price relative to weekly EMA200
        is_uptrend = close[i] > ema_200_1w_aligned[i]
        is_downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Entry conditions: price touches Camarilla S3/R3 with volume and trend alignment
        # Use small buffer to avoid whipsaws
        buffer = 0.001 * close[i]  # 0.1% buffer
        
        long_entry = (close[i] <= s3_aligned[i] + buffer) and volume_filter and is_uptrend
        short_entry = (close[i] >= r3_aligned[i] - buffer) and volume_filter and is_downtrend
        
        # Exit conditions: return to pivot or trend reversal
        long_exit = (close[i] >= pivot_aligned[i]) or (not is_uptrend)
        short_exit = (close[i] <= pivot_aligned[i]) or (not is_downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals