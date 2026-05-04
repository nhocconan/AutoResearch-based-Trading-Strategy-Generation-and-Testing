#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R3/S3 Breakout + 1w EMA50 Trend Filter + Volume Spike Confirmation
# Camarilla pivots provide intraday support/resistance levels derived from previous day's range.
# Breakout above R3 or below S3 with volume spike indicates strong momentum.
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via long signals in uptrend and bear markets via short signals in downtrend.

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to access previous day's data
        # Skip if any value is NaN
        if np.isnan(ema_50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Need previous day's OHLC for Camarilla calculation
        if i < len(df_1d):
            prev_high = df_1d['high'].iloc[i-1]
            prev_low = df_1d['low'].iloc[i-1]
            prev_close = df_1d['close'].iloc[i-1]
        else:
            # Use last available day if we've exceeded 1d data length
            prev_high = df_1d['high'].iloc[-1]
            prev_low = df_1d['low'].iloc[-1]
            prev_close = df_1d['close'].iloc[-1]
        
        # Calculate Camarilla levels for current day
        # Camarilla formula: 
        # R4 = Close + (High-Low) * 1.1/2
        # R3 = Close + (High-Low) * 1.1/4
        # R2 = Close + (High-Low) * 1.1/6
        # R1 = Close + (High-Low) * 1.1/12
        # PP = (High + Low + Close)/3
        # S1 = Close - (High-Low) * 1.1/12
        # S2 = Close - (High-Low) * 1.1/6
        # S3 = Close - (High-Low) * 1.1/4
        # S4 = Close - (High-Low) * 1.1/2
        
        range_hl = prev_high - prev_low
        r3 = prev_close + range_hl * 1.1 / 4
        s3 = prev_close - range_hl * 1.1 / 4
        
        # Volume spike filter (20-period volume EMA on 12h data)
        if i >= 20:
            vol_ema_20 = pd.Series(volume[:i+1]).ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
            volume_spike = volume[i] > (vol_ema_20 * 2.0)
        else:
            volume_spike = False
        
        if position == 0:
            # Long conditions: Close above R3 AND weekly uptrend AND volume spike
            if (close[i] > r3 and 
                close[i] > ema_50_aligned[i] and  # 1w uptrend
                volume_spike):
                signals[i] = 0.30
                position = 1
            # Short conditions: Close below S3 AND weekly downtrend AND volume spike
            elif (close[i] < s3 and 
                  close[i] < ema_50_aligned[i] and  # 1w downtrend
                  volume_spike):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Price closes below R3 OR weekly trend turns down
            if (close[i] < r3 or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Price closes above S3 OR weekly trend turns up
            if (close[i] > s3 or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals