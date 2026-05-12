#!/usr/bin/env python3
# 1H_CAMARILLA_PIVOT_REVERSION_4H_TREND_FILTER
# Hypothesis: Camarilla pivot reversals (S1/S3 for long, R1/R3 for short) work best when aligned with 4h trend.
# In 4h uptrend (price > EMA50), look for longs at S1/S3; in downtrend, shorts at R1/R3.
# Uses volume confirmation to avoid false breakouts. Session filter (08-20 UTC) reduces noise.
# Target: 20-40 trades/year on 1h timeframe.

name = "1H_CAMARILLA_PIVOT_REVERSION_4H_TREND_FILTER"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter and pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # EMA50 for 4h trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivots from previous 4h bar
    # Typical price = (H + L + C) / 3
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    range_4h = df_4h['high'] - df_4h['low']
    
    # Camarilla levels
    S1 = typical_price - (range_4h * 1.0 / 6)
    S2 = typical_price - (range_4h * 2.0 / 6)
    S3 = typical_price - (range_4h * 3.0 / 6)
    R1 = typical_price + (range_4h * 1.0 / 6)
    R2 = typical_price + (range_4h * 2.0 / 6)
    R3 = typical_price + (range_4h * 3.0 / 6)
    
    # Align 4h data to 1h
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1.values)
    S2_aligned = align_htf_to_ltf(prices, df_4h, S2.values)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3.values)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1.values)
    R2_aligned = align_htf_to_ltf(prices, df_4h, R2.values)
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3.values)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if session_ok[i] and volume_ok[i]:
            if position == 0:
                # LONG: 4h uptrend + price at S1 or S3 (reversal from support)
                if (close[i] > ema50_4h_aligned[i] and 
                    (close[i] <= S1_aligned[i] * 1.005 or close[i] <= S3_aligned[i] * 1.005)):
                    signals[i] = 0.20
                    position = 1
                # SHORT: 4h downtrend + price at R1 or R3 (reversal from resistance)
                elif (close[i] < ema50_4h_aligned[i] and 
                      (close[i] >= R1_aligned[i] * 0.995 or close[i] >= R3_aligned[i] * 0.995)):
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: Trend reversal or price reaches opposite resistance
                if (close[i] <= ema50_4h_aligned[i] or 
                    close[i] >= R1_aligned[i] * 0.995):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # EXIT SHORT: Trend reversal or price reaches opposite support
                if (close[i] >= ema50_4h_aligned[i] or 
                    close[i] <= S1_aligned[i] * 1.005):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # Outside session or low volume: flatten
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals