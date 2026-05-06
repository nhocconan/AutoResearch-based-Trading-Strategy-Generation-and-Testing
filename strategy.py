#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d structure with volume confirmation
# Long when: price > 4h EMA50 (trend filter) AND price breaks above 1d Camarilla R1 level with volume spike
# Short when: price < 4h EMA50 (trend filter) AND price breaks below 1d Camarilla S1 level with volume spike
# Exit: price returns to 1d Camarilla midpoint (PP) or opposite Camarilla level (R3/S3)
# Uses discrete sizing 0.20 to balance return and drawdown control
# Target: 60-150 total trades over 4 years = 15-37/year for 1h
# Session filter: 08-20 UTC to reduce noise trades
# Volume confirmation: 1h volume > 1.5 * 20-period average volume
# Works in both bull (continuation breakouts) and bear (continuation breakdowns) markets

name = "1h_4hEMA50_1dCamarilla_R1S1_Breakout_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC) - ONCE before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    close_4h = df_4h['close'].values
    # Calculate 4h EMA50
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 completed daily bars for Camarilla
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R1 = PP + (High - Low) * 1.1/12
    # S1 = PP - (High - Low) * 1.1/12
    # R3 = PP + (High - Low) * 1.1/4
    # S3 = PP - (High - Low) * 1.1/4
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = pp + (high_1d - low_1d) * 1.1 / 12.0
    s1 = pp - (high_1d - low_1d) * 1.1 / 12.0
    r3 = pp + (high_1d - low_1d) * 1.1 / 4.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align 1d Camarilla levels to 1h timeframe (wait for completed 1d bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > 4h EMA50 (uptrend) AND price breaks above 1d Camarilla R1 level with volume spike
            if (close[i] > ema_50_4h_aligned[i] and 
                close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price < 4h EMA50 (downtrend) AND price breaks below 1d Camarilla S1 level with volume spike
            elif (close[i] < ema_50_4h_aligned[i] and 
                  close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to 1d Camarilla PP or breaks below S3 (strong reversal)
            if close[i] <= pp_aligned[i] or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to 1d Camarilla PP or breaks above R3 (strong reversal)
            if close[i] >= pp_aligned[i] or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals