#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R4/S4 breakout with 12h EMA trend filter and volume confirmation.
Long when price breaks above R4 and 12h EMA50 > EMA200 with volume > 1.5x average.
Short when price breaks below S4 and 12h EMA50 < EMA200 with volume > 1.5x average.
Exit on opposite Camarilla level break or EMA crossover reversal.
Uses 12h EMA for trend filter (more responsive than 1d ADX) and R4/S4 for stronger breakouts.
Designed for 6h timeframe targeting 75-200 total trades over 4 years with volume confirmation to reduce false signals.
Works in both bull and bear markets by only taking breakouts in direction of 12h EMA trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMAs on 12h data
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMAs to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200)
    
    # Calculate Camarilla levels from prior 12h bar
    def calculate_camarilla(high, low, close):
        # Typical price for pivot
        pivot = (high + low + close) / 3
        range_val = high - low
        
        # Camarilla levels
        r3 = pivot + (range_val * 1.1 / 4)
        s3 = pivot - (range_val * 1.1 / 4)
        r4 = pivot + (range_val * 1.1 / 2)
        s4 = pivot - (range_val * 1.1 / 2)
        
        return r3, s3, r4, s4
    
    # Calculate Camarilla levels for each 12h bar (using prior bar's data)
    camarilla_r4 = np.full(len(close_12h), np.nan)
    camarilla_s4 = np.full(len(close_12h), np.nan)
    
    for i in range(1, len(close_12h)):
        _, _, r4, s4 = calculate_camarilla(df_12h['high'].values[i-1], df_12h['low'].values[i-1], close_12h[i-1])
        camarilla_r4[i] = r4
        camarilla_s4[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_aligned[i]
        ema_200_val = ema_200_aligned[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above R4 AND 12h EMA50 > EMA200 (uptrend) AND volume spike
            if (price > r4_val and ema_50_val > ema_200_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S4 AND 12h EMA50 < EMA200 (downtrend) AND volume spike
            elif (price < s4_val and ema_50_val < ema_200_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below S4 OR EMA50 < EMA200 (trend reversal)
                if (price < s4_val or ema_50_val < ema_200_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above R4 OR EMA50 > EMA200 (trend reversal)
                if (price > r4_val or ema_50_val > ema_200_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R4_S4_Breakout_12hEMA_Volume"
timeframe = "6h"
leverage = 1.0