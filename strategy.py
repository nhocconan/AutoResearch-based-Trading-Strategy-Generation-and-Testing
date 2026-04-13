#!/usr/bin/env python3
"""
12h_1w_Camarilla_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: Price rejection at Camarilla pivot levels on 12h timeframe with weekly trend filter and volume confirmation captures swing reversals.
Works in bull markets via buying dips at S1/S2 with weekly uptrend, and in bear markets via selling rallies at R1/R2 with weekly downtrend.
Targets 15-25 trades/year per symbol by requiring confluence of pivot level, volume spike, and weekly trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Calculate Camarilla pivot levels from previous day
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R4 = Close + ((High-Low) * 1.5000)
    # R3 = Close + ((High-Low) * 1.2500)
    # R2 = Close + ((High-Low) * 1.1666)
    # R1 = Close + ((High-Low) * 1.0833)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High-Low) * 1.0833)
    # S2 = Close - ((High-Low) * 1.1666)
    # S3 = Close - ((High-Low) * 1.2500)
    # S4 = Close - ((High-Low) * 1.5000)
    
    high_low_diff = prev_high - prev_low
    r4 = prev_close + (high_low_diff * 1.5000)
    r3 = prev_close + (high_low_diff * 1.2500)
    r2 = prev_close + (high_low_diff * 1.1666)
    r1 = prev_close + (high_low_diff * 1.0833)
    pp = (prev_high + prev_low + prev_close) / 3.0
    s1 = prev_close - (high_low_diff * 1.0833)
    s2 = prev_close - (high_low_diff * 1.1666)
    s3 = prev_close - (high_low_diff * 1.2500)
    s4 = prev_close - (high_low_diff * 1.5000)
    
    # Weekly EMA trend filter (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        ema21_1w = np.full(len(prices), np.nan)
    else:
        close_1w = df_1w['close'].values
        ema21_1w_raw = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema21_1w = align_htf_to_ltf(prices, df_1w, ema21_1w_raw)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(40, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(r1[i]) or np.isnan(r2[i]) or np.isnan(s1[i]) or np.isnan(s2[i]) or 
            np.isnan(ema21_1w[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long setup: price near S1 or S2 with weekly uptrend and volume expansion
        near_s1 = low[i] <= s1[i] * 1.002 and low[i] >= s1[i] * 0.998  # within 0.2% of S1
        near_s2 = low[i] <= s2[i] * 1.002 and low[i] >= s2[i] * 0.998  # within 0.2% of S2
        near_support = near_s1 or near_s2
        weekly_uptrend = close[i] > ema21_1w[i]
        
        long_signal = near_support and weekly_uptrend and volume_expansion[i]
        
        # Short setup: price near R1 or R2 with weekly downtrend and volume expansion
        near_r1 = high[i] >= r1[i] * 0.998 and high[i] <= r1[i] * 1.002  # within 0.2% of R1
        near_r2 = high[i] >= r2[i] * 0.998 and high[i] <= r2[i] * 1.002  # within 0.2% of R2
        near_resistance = near_r1 or near_r2
        weekly_downtrend = close[i] < ema21_1w[i]
        
        short_signal = near_resistance and weekly_downtrend and volume_expansion[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1w_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0