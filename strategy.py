#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above R3 with volume > 1.5x 20-bar average and close > 4h EMA50 (uptrend)
# Short when price breaks below S3 with volume > 1.5x 20-bar average and close < 4h EMA50 (downtrend)
# Exit when price retests the Camarilla pivot level (middle line) or trend fails (close crosses 4h EMA50)
# Camarilla levels provide precise intraday support/resistance. Works in bull (buy breakouts) and bear (sell breakdowns).
# Target: 60-150 total trades over 4 years = 15-37/year. Uses discrete sizing (0.20) to minimize fee churn.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC (pre-compute before loop)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for 1d (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2 (same as H4/L4)
    camarilla_range = 1.1 * (high_1d - low_1d) / 2
    r3_level = close_1d + camarilla_range  # R3 and H4 are identical
    s3_level = close_1d - camarilla_range  # S3 and L4 are identical
    pivot_level = (high_1d + low_1d + close_1d) / 3  # Camarilla pivot (middle line)
    
    # Align 1d levels to 1h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_level)
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 20) + 1  # EMA50(4h) + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_50_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume spike and close > 4h EMA50 (uptrend)
            if (close[i] > r3_aligned[i] and 
                volume_spike[i] and close[i] > ema_50_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 with volume spike and close < 4h EMA50 (downtrend)
            elif (close[i] < s3_aligned[i] and 
                  volume_spike[i] and close[i] < ema_50_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price retests pivot level or close < 4h EMA50 (trend failure)
            if (close[i] <= pivot_aligned[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price retests pivot level or close > 4h EMA50 (trend failure)
            if (close[i] >= pivot_aligned[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals