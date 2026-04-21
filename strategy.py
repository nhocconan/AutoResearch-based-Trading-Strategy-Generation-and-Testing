#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power with daily EMA50 trend filter and volume confirmation.
Long when Bull Power > 0 (close > EMA13) and Bear Power < 0 (low < EMA13) with EMA50 uptrend and volume > 1.3x average.
Short when Bear Power < 0 (low < EMA13) and Bull Power < 0 (close < EMA13) with EMA50 downtrend and volume > 1.3x average.
Exit when power signals reverse or volume drops below average.
Uses Elder Ray to measure bull/bear power relative to EMA13, EMA50 for trend, and volume for confirmation.
Designed for 15-25 trades/year to minimize fee drift while capturing sustained moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Elder Ray and EMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily price arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Close - EMA13
    bull_power = close_1d - ema13
    
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema13
    
    # EMA50 for trend filter
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current daily values aligned to 6h
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_1d_current = vol_1d_aligned[i]
        
        # Close price for reference
        price_close = prices['close'].iloc[i]
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power < 0, EMA50 uptrend, volume surge
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and
                ema50_aligned[i] > ema50_aligned[max(i-1, 0)] and  # EMA50 rising
                vol_1d_current > 1.3 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0, Bull Power < 0, EMA50 downtrend, volume surge
            elif (bear_power_aligned[i] < 0 and 
                  bull_power_aligned[i] < 0 and
                  ema50_aligned[i] < ema50_aligned[max(i-1, 0)] and  # EMA50 falling
                  vol_1d_current > 1.3 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: power signals reverse or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power <= 0 OR Bear Power >= 0 OR volume < average
                if (bull_power_aligned[i] <= 0 or 
                    bear_power_aligned[i] >= 0 or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Bull Power >= 0 OR Bear Power >= 0 OR volume < average
                if (bull_power_aligned[i] >= 0 or 
                    bear_power_aligned[i] >= 0 or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_Power_EMA50_Trend_Volume1.3x"
timeframe = "6h"
leverage = 1.0