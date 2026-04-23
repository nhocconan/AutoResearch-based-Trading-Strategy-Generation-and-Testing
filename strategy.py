#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R1 (12h Camarilla) AND price > 1d EMA34 (uptrend) AND volume > 1.8x average.
Short when price breaks below S1 (12h Camarilla) AND price < 1d EMA34 (downtrend) AND volume > 1.8x average.
Exit when price reverts to Camarilla pivot point (PP) or trend reverses (price crosses 1d EMA34).
Uses 12h timeframe for lower trade frequency (target: 50-150 total over 4 years) with tight entry conditions.
1d EMA34 provides stable trend filter. Volume confirmation ensures high-conviction breakouts.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
Target: 50-150 trades over 4 years (12-37/year).
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
    
    # Load 12h data for Camarilla pivot levels - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Camarilla pivot levels for 12h
    # Camarilla: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_pp = (high_12h + low_12h + close_12h) / 3.0
    camarilla_r1 = close_12h + (high_12h - low_12h) * 1.1 / 12.0
    camarilla_s1 = close_12h - (high_12h - low_12h) * 1.1 / 12.0
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h indicators to 12h timeframe (no alignment needed for same timeframe)
    # Align 1d EMA34 to 12h timeframe
    camarilla_pp_aligned = camarilla_pp  # no alignment needed for same timeframe
    camarilla_r1_aligned = camarilla_r1
    camarilla_s1_aligned = camarilla_s1
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (30-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        pp_val = camarilla_pp_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > r1_val and price > ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < s1_val and price < ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to pivot point OR price breaks below 1d EMA34 (trend reversal)
                if price <= pp_val or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to pivot point OR price breaks above 1d EMA34 (trend reversal)
                if price >= pp_val or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1_S1_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0