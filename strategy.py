#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA trend filter and volume spike confirmation.
Long when price breaks above R1, 1d EMA34 > 1d EMA89, and volume > 2x 20-period average.
Short when price breaks below S1, 1d EMA34 < 1d EMA89, and volume > 2x average.
Exit when price retests the Camarilla pivot (PP) or volume drops below average.
Uses Camarilla levels for intraday support/resistance, EMA for trend filter, and volume for confirmation.
Designed for 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Pivot point (PP)
    PP = (high_prev + low_prev + close_prev) / 3.0
    
    # Camarilla levels
    R1 = close_prev + 1.1 * (high_prev - low_prev) / 12.0
    S1 = close_prev - 1.1 * (high_prev - low_prev) / 12.0
    R4 = close_prev + 1.1 * (high_prev - low_prev) / 2.0
    S4 = close_prev - 1.1 * (high_prev - low_prev) / 2.0
    
    # Calculate daily EMAs for trend filter
    close_1d = df_1d['close'].values
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89 = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Calculate daily volume average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    ema89_aligned = align_htf_to_ltf(prices, df_1d, ema89)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(ema89_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(vol_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        
        # Current values
        vol_1d_current = vol_1d_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above R1, EMA34 > EMA89, volume spike
            if (price_high > R1_aligned[i] and 
                ema34_aligned[i] > ema89_aligned[i] and
                vol_1d_current > 2.0 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1, EMA34 < EMA89, volume spike
            elif (price_low < S1_aligned[i] and 
                  ema34_aligned[i] < ema89_aligned[i] and
                  vol_1d_current > 2.0 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price retests PP or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches or goes below PP OR volume < average
                if price_low <= PP_aligned[i]:
                    exit_signal = True
                elif vol_1d_current < vol_ma_20_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price touches or goes above PP OR volume < average
                if price_high >= PP_aligned[i]:
                    exit_signal = True
                elif vol_1d_current < vol_ma_20_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_89_Trend_Volume2x"
timeframe = "4h"
leverage = 1.0