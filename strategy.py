#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA200 trend filter and 1d ATR volatility filter.
Long when price breaks above 1h Camarilla R1 level AND price > 4h EMA200 (uptrend) AND 1d ATR(14) > 0.5 * 1h ATR(14) (high volatility regime).
Short when price breaks below 1h Camarilla S1 level AND price < 4h EMA200 (downtrend) AND 1d ATR(14) > 0.5 * 1h ATR(14).
Exit when price reverts to 1h Camarilla pivot point (PP) or 4h EMA200 trend reverses.
Uses 1h timeframe with tight entry conditions (Camarilla R1/S1 are strong resistance/support) to limit trades.
4h EMA200 provides strong trend filter. 1d ATR regime filter ensures trading only in sufficient volatility.
Target: 60-150 trades over 4 years (15-37/year) to stay within proven working range for 1h.
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
    
    # Calculate 1h Camarilla levels (R1, S1, PP) - ONCE before loop
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 10:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Camarilla levels on 1h (based on previous 1h bar's OHLC)
    prev_high_1h = np.roll(high_1h, 1)
    prev_low_1h = np.roll(low_1h, 1)
    prev_close_1h = np.roll(close_1h, 1)
    # Set first value to NaN (no previous bar)
    prev_high_1h[0] = np.nan
    prev_low_1h[0] = np.nan
    prev_close_1h[0] = np.nan
    
    camarilla_pp = (prev_high_1h + prev_low_1h + prev_close_1h) / 3.0
    camarilla_r1 = prev_close_1h + (prev_high_1h - prev_low_1h) * 1.1 / 12.0
    camarilla_s1 = prev_close_1h - (prev_high_1h - prev_low_1h) * 1.1 / 12.0
    
    # Load 4h data for EMA200 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Load 1d data for ATR(14) volatility filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # True Range for 1h (for volatility comparison)
    tr1_h = high - low
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = np.nan
    tr2_h[0] = np.nan
    tr3_h[0] = np.nan
    tr_1h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr_1h = pd.Series(tr_1h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s1)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema200_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(atr_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pp_val = camarilla_pp_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        ema200_val = ema200_4h_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_1h_val = atr_1h[i]
        
        # Get current price
        price = close[i]
        
        # Volatility filter: 1d ATR > 0.5 * 1h ATR (ensures sufficient volatility)
        vol_filter = atr_1d_val > 0.5 * atr_1h_val
        
        if position == 0:
            # Long: price breaks above 1h Camarilla R1 AND price > 4h EMA200 (uptrend) AND volatility filter
            if (price > r1_val and price > ema200_val and vol_filter):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 1h Camarilla S1 AND price < 4h EMA200 (downtrend) AND volatility filter
            elif (price < s1_val and price < ema200_val and vol_filter):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to Camarilla PP OR price breaks below 4h EMA200 (trend reversal)
                if price <= pp_val or price < ema200_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to Camarilla PP OR price breaks above 4h EMA200 (trend reversal)
                if price >= pp_val or price > ema200_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1_S1_Breakout_4hEMA200_1dATR_VolFilter"
timeframe = "1h"
leverage = 1.0