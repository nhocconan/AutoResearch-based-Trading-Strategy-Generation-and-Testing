#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R4/S4 Breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above R4 AND price > 12h EMA50 AND volume > 2x average.
Short when price breaks below S4 AND price < 12h EMA50 AND volume > 2x average.
Exit when price retouches the Camarilla pivot point (PP) or volume drops below average.
Camarilla levels from 1d provide intraday support/resistance. R4/S4 breaks indicate strong momentum.
12h EMA50 ensures trading in direction of intermediate trend.
Volume confirmation avoids false breakouts.
Designed for 6h timeframe targeting 50-150 total trades over 4 years with low frequency to minimize fee drag.
Works in both bull and bear markets by only taking breakouts aligned with 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (based on previous day)
    # We'll calculate for the current 1d bar using previous day's high/low/close
    camarilla_pp = np.full(len(close_1d), np.nan)
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's data to calculate today's levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3
        range_ = prev_high - prev_low
        
        camarilla_pp[i] = pivot
        camarilla_r4[i] = pivot + (range_ * 1.5)
        camarilla_s4[i] = pivot - (range_ * 1.5)
    
    # Align 1d Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pp_val = pp_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        ema50_val = ema50_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Price breaks above R4 AND price > 12h EMA50 AND volume spike
            if (price > r4_val and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S4 AND price < 12h EMA50 AND volume spike
            elif (price < s4_val and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price retouches pivot point OR volume drops below average
                if (price <= pp_val or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price retouches pivot point OR volume drops below average
                if (price >= pp_val or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R4_S4_Breakout_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0