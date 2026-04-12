# 1d_1w_camarilla_breakout_with_volume_and_trend
# Hypothesis: Daily Camarilla breakout with weekly trend filter and volume confirmation
# Uses weekly EMA to filter direction (long only in uptrend, short only in downtrend)
# Daily Camarilla levels from previous day for entry/exit
# Volume > 1.5x 20-day average for confirmation
# Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drift

name = "1d_1w_camarilla_breakout_with_volume_and_trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(20) for trend filter
    ema_20w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    # Get daily data for Camarilla and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range (for today's Camarilla levels)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Camarilla levels (based on previous day)
    range_ = prev_high - prev_low
    # Resistance levels
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    # Support levels
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Align Camarilla levels to daily timeframe (no shift needed as they're based on prev day)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_20w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price > weekly EMA20 (uptrend) + break above R4 + volume
        if (close[i] > ema_20w_aligned[i] and 
            close[i] > r4_aligned[i] and 
            vol_confirm[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short conditions: price < weekly EMA20 (downtrend) + break below S4 + volume
        elif (close[i] < ema_20w_aligned[i] and 
              close[i] < s4_aligned[i] and 
              vol_confirm[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals