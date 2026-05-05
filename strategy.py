#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout + 1d EMA34 trend + volume spike
# Long when: price breaks above R3 (1d) AND close > EMA34(1d) AND volume > 2x 20-period MA
# Short when: price breaks below S3 (1d) AND close < EMA34(1d) AND volume > 2x 20-period MA
# Exit when: price returns to Camarilla pivot (PP) OR volume drops below average
# Uses Camarilla for institutional levels, EMA for trend filter, volume for conviction
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations (using previous day's range)
    PP = np.zeros(len(close_1d))
    R3 = np.zeros(len(close_1d))
    S3 = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        # Pivot point
        PP[i] = (prev_high + prev_low + prev_close) / 3.0
        
        # Camarilla levels
        range_prev = prev_high - prev_low
        R3[i] = PP[i] + range_prev * 1.1 / 4.0
        S3[i] = PP[i] - range_prev * 1.1 / 4.0
    
    # For first bar, use same values (will be aligned properly)
    PP[0] = PP[1] if len(PP) > 1 else close_1d[0]
    R3[0] = R3[1] if len(R3) > 1 else close_1d[0]
    S3[0] = S3[1] if len(S3) > 1 else close_1d[0]
    
    # Calculate 1d EMA34
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)  # 2x volume spike
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(PP_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3 + above EMA34 + volume spike
            if (close[i] > R3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S3 + below EMA34 + volume spike
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to pivot OR volume drops below average
            if (close[i] <= PP_aligned[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to pivot OR volume drops below average
            if (close[i] >= PP_aligned[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals