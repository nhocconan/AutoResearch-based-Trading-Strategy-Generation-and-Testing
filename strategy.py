# 4h_Camarilla_Pivot_S1_S3_Breakout_12hTrend_Volume
# Hypothesis: 4h Camarilla pivot breakout with 12h trend filter and volume confirmation
# Works in bull markets via breakout above resistance and in bear via breakdown below support
# Volume ensures genuine breakouts, 12h EMA50 filters counter-trend moves
# Target: 20-50 trades/year with size 0.25 to minimize fee drag

name = "4h_Camarilla_Pivot_S1_S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels (S1, S3, R1, R3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot and Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    S1 = prev_close - (range_val * 1.0 / 6.0)
    S3 = prev_close - (range_val * 3.0 / 6.0)
    R1 = prev_close + (range_val * 1.0 / 6.0)
    R3 = prev_close + (range_val * 3.0 / 6.0)
    
    # Align Camarilla levels to 4h timeframe
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(S1_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_surge[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 with volume surge and above 12h EMA50
            if (close[i] > R1_aligned[i]) and volume_surge[i] and (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with volume surge and below 12h EMA50
            elif (close[i] < S1_aligned[i]) and volume_surge[i] and (close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 (strong reversal) or volume drops
            if (close[i] < S3_aligned[i]) or (not volume_surge[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 (strong reversal) or volume drops
            if (close[i] > R3_aligned[i]) or (not volume_surge[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals