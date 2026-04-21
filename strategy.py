#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and 1d RSI trend filter.
Long when price breaks above R1 with volume > 1.3x average and daily RSI > 50.
Short when price breaks below S1 with volume > 1.3x average and daily RSI < 50.
Exit when price returns to daily pivot or volume drops below average.
Camarilla levels provide intraday support/resistance, volume confirms breakout strength,
and RSI filters for trend alignment to avoid counter-trend trades in chop.
Target: 20-30 trades/year for low fee drag and robust performance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla levels, RSI, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily pivot and Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + 1.1*range/12, S1 = close - 1.1*range/12
    r1 = close_1d + 1.1 * range_1d / 12
    s1 = close_1d - 1.1 * range_1d / 12
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # prepend NaN for first element
    
    # Align daily RSI to 4h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate daily volume average (14-period)
    vol_1d = df_1d['volume'].values
    vol_ma_14 = pd.Series(vol_1d).rolling(window=14, min_periods=14).mean().values
    vol_ma_14_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price_close = prices['close'].iloc[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)[i]
        
        if position == 0:
            # Enter long: price breaks above R1, volume surge, daily RSI > 50 (uptrend)
            if (price_close > r1_aligned[i] and 
                vol_1d_current > 1.3 * vol_ma_14_aligned[i] and
                rsi_aligned[i] > 50):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1, volume surge, daily RSI < 50 (downtrend)
            elif (price_close < s1_aligned[i] and 
                  vol_1d_current > 1.3 * vol_ma_14_aligned[i] and
                  rsi_aligned[i] < 50):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to daily pivot or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= pivot or volume < average
                if (price_close <= pivot_aligned[i] or
                    vol_1d_current < vol_ma_14_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price >= pivot or volume < average
                if (price_close >= pivot_aligned[i] or
                    vol_1d_current < vol_ma_14_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Volume1.3x_DailyRSI50"
timeframe = "4h"
leverage = 1.0