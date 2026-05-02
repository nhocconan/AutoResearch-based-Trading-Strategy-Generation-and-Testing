#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Uses discrete sizing (0.25) to minimize fee drag while capturing institutional breakouts
# Volume spike (2.0x 24-bar MA) confirms participation
# Trend filter ensures alignment with weekly direction
# Designed for 30-100 total trades over 4 years (7-25/year) on 1d timeframe
# Works in bull markets (breakouts with trend) and bear markets (mean reversion at extremes)

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w OHLC for Camarilla levels (Pivot-based)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    # Camarilla levels: R3, R4, S3, S4
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # R4 = Close + Range * 1.1/2
    # R3 = Close + Range * 1.1/4
    # S3 = Close - Range * 1.1/4
    # S4 = Close - Range * 1.1/2
    pivot_1w = (high_1w + low_1w + close_1w_arr) / 3.0
    range_1w = high_1w - low_1w
    r3_1w = close_1w_arr + range_1w * 1.1 / 4.0
    r4_1w = close_1w_arr + range_1w * 1.1 / 2.0
    s3_1w = close_1w_arr - range_1w * 1.1 / 4.0
    s4_1w = close_1w_arr - range_1w * 1.1 / 2.0
    
    # Align Camarilla levels to 1d timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Volume confirmation: 2.0x 24-period average (~1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50  # EMA50 warmup
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above R3 AND price > 1w EMA50 (uptrend bias) AND volume spike
            if (close[i] > r3_1w_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S3 AND price < 1w EMA50 (downtrend bias) AND volume spike
            elif (close[i] < s3_1w_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S3 OR price below 1w EMA50 (trend failure)
            if close[i] < s3_1w_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R3 OR price above 1w EMA50 (trend failure)
            if close[i] > r3_1w_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals