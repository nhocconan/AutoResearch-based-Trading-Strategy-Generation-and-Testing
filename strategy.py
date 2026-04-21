#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot breakout with 12h volume spike and EMA trend filter.
Longs when price breaks above R1 with EMA(50) upward and volume > 1.5x average; shorts when price breaks below S1 with EMA(50) downward and volume > 1.5x average.
Exit on price crossing back through pivot point (PP) or 1.5x ATR stop.
Designed for 25-35 trades/year to minimize fee drag while capturing high-probability breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day's range
    range_1d = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3
    camarilla_r1 = camarilla_pp + 1.1 * range_1d / 12
    camarilla_s1 = camarilla_pp - 1.1 * range_1d / 12
    
    # Calculate 50-period EMA for trend filter (using daily close)
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Camarilla levels and EMA to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Load 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume average on 12h timeframe
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h volume average to 4h timeframe
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # ATR for stoploss (20-period on 4h data)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        pp = camarilla_pp_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_ma_20_12h_val = vol_ma_20_12h_aligned[i]
        atr_val = atr[i]
        
        # Current 12h volume (need to align raw 12h volume to 4h)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
        vol_12h_current = vol_12h_aligned[i]
        
        if position == 0:
            # Enter long: break above R1 with upward EMA and volume spike
            if (price_high > r1 and 
                ema_val > ema_50_aligned[i-1] and  # EMA rising
                vol_12h_current > 1.5 * vol_ma_20_12h_val):
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with downward EMA and volume spike
            elif (price_low < s1 and 
                  ema_val < ema_50_aligned[i-1] and  # EMA falling
                  vol_12h_current > 1.5 * vol_ma_20_12h_val):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: pivot point cross OR ATR-based stoploss
            exit_signal = False
            
            # Pivot point exit
            if position == 1 and price_close < pp:
                exit_signal = True
            elif position == -1 and price_close > pp:
                exit_signal = True
            
            # ATR-based stoploss (1.5x ATR from entry level)
            if position == 1:
                # For longs, stop below entry area (using S1 as reference)
                if price_close < s1 - 1.5 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above entry area (using R1 as reference)
                if price_close > r1 + 1.5 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hVolSpike_EMA50Trend_ATR1.5x"
timeframe = "4h"
leverage = 1.0