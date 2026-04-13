# 12h_1d_1w_Camarilla_Pivot_Breakout_Volume
# Hypothesis: Combines Camarilla pivot levels from daily with weekly trend filter and volume confirmation on 12h.
# In bull markets, buys near S3/S4 with stop at S5; in bear markets, sells near R3/R4 with stop at R5.
# Uses weekly trend to filter direction (only long when weekly close > weekly open, short when weekly close < weekly open).
# Volume confirmation requires 12h volume > 1.5x 20-period average.
# Target: 15-37 trades/year on 12h (60-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    close_val = close
    S1 = close - (range_val * 1.1 / 12)
    S2 = close - (range_val * 1.1 / 6)
    S3 = close - (range_val * 1.1 / 4)
    S4 = close - (range_val * 1.1 / 2)
    R1 = close + (range_val * 1.1 / 12)
    R2 = close + (range_val * 1.1 / 6)
    R3 = close + (range_val * 1.1 / 4)
    R4 = close + (range_val * 1.1 / 2)
    return S1, S2, S3, S4, R1, R2, R3, R4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar
    S1_1d = np.full(len(close_1d), np.nan)
    S2_1d = np.full(len(close_1d), np.nan)
    S3_1d = np.full(len(close_1d), np.nan)
    S4_1d = np.full(len(close_1d), np.nan)
    R1_1d = np.full(len(close_1d), np.nan)
    R2_1d = np.full(len(close_1d), np.nan)
    R3_1d = np.full(len(close_1d), np.nan)
    R4_1d = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        S1, S2, S3, S4, R1, R2, R3, R4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        S1_1d[i] = S1
        S2_1d[i] = S2
        S3_1d[i] = S3
        S4_1d[i] = S4
        R1_1d[i] = R1
        R2_1d[i] = R2
        R3_1d[i] = R3
        R4_1d[i] = R4
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    
    # Weekly trend: 1 = bullish (close > open), -1 = bearish (close < open)
    weekly_trend = np.where(close_1w > open_1w, 1, -1)
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period volume average on 12h
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean()
    volume_expansion_12h = volume_12h > (vol_ma_20_12h * 1.5)
    
    # Align all signals to 12h timeframe (which is our primary timeframe)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1_1d)
    S2_12h = align_htf_to_ltf(prices, df_1d, S2_1d)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_12h = align_htf_to_ltf(prices, df_1d, S4_1d)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1_1d)
    R2_12h = align_htf_to_ltf(prices, df_1d, R2_1d)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3_1d)
    R4_12h = align_htf_to_ltf(prices, df_1d, R4_1d)
    weekly_trend_12h = align_htf_to_ltf(prices, df_1w, weekly_trend)
    volume_expansion_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_expansion_12h)
    
    # Session filter: 00:00-23:00 UTC (trade all hours for 12h)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(S3_12h[i]) or np.isnan(S4_12h[i]) or \
           np.isnan(R3_12h[i]) or np.isnan(R4_12h[i]) or \
           np.isnan(weekly_trend_12h[i]) or \
           np.isnan(volume_expansion_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions: weekly bullish + price at S3/S4 level + volume expansion
        if (weekly_trend_12h[i] == 1 and 
            volume_expansion_12h_aligned[i] and
            ((abs(close[i] - S3_12h[i]) / close[i] < 0.005) or  # Near S3
             (abs(close[i] - S4_12h[i]) / close[i] < 0.005))):  # Near S4
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        
        # Short conditions: weekly bearish + price at R3/R4 level + volume expansion
        elif (weekly_trend_12h[i] == -1 and 
              volume_expansion_12h_aligned[i] and
              ((abs(close[i] - R3_12h[i]) / close[i] < 0.005) or  # Near R3
               (abs(close[i] - R4_12h[i]) / close[i] < 0.005))):  # Near R4
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        
        # Hold current position
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        
        # Exit conditions: price reaches S5/R5 levels or weekly trend changes
        elif position == 1:
            # Exit long if price reaches S5 or weekly trend turns bearish
            S5_12h = S4_12h[i] - ((R4_12h[i] - S4_12h[i]) * 1.1 / 2)  # S5 = S4 - (R4-S4)*1.1/2
            if close[i] <= S5_12h or weekly_trend_12h[i] == -1:
                position = 0
                signals[i] = 0.0
        
        elif position == -1:
            # Exit short if price reaches R5 or weekly trend turns bullish
            R5_12h = R4_12h[i] + ((R4_12h[i] - S4_12h[i]) * 1.1 / 2)  # R5 = R4 + (R4-S4)*1.1/2
            if close[i] >= R5_12h or weekly_trend_12h[i] == 1:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0