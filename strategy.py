#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R4 AND close > 12h EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below Camarilla S4 AND close < 12h EMA50 AND volume > 1.8x 20-period average.
Exit when price retraces to Camarilla pivot point (PP) or ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) to reduce fee drag and improve test generalization.
Camarilla R4/S4 levels provide stronger breakout confirmation than R3/S3.
12h EMA50 filter ensures alignment with medium-term trend, reducing whipsaws in choppy markets.
Volume spike confirms institutional participation. Works in bull markets (breakouts with volume in uptrend)
and bear markets (breakdowns with volume in downtrend) by following the 12h trend.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Camarilla levels from 1d OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculation: PP = (H+L+C)/3, R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_pp_1d = typical_price_1d
    camarilla_r4_1d = close_1d + (range_1d * 1.1)
    camarilla_s4_1d = close_1d - (range_1d * 1.1)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp_1d)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation (using 4h data)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_4h_val = atr_4h[i]
        ema50_val = ema50_12h_aligned[i]
        camarilla_pp = camarilla_pp_aligned[i]
        camarilla_r4 = camarilla_r4_aligned[i]
        camarilla_s4 = camarilla_s4_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R4 AND uptrend (price > EMA50) AND volume spike
            if close[i] > camarilla_r4 and close[i] > ema50_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S4 AND downtrend (price < EMA50) AND volume spike
            elif close[i] < camarilla_s4 and close[i] < ema50_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to Camarilla pivot point
            if position == 1 and close[i] <= camarilla_pp:
                exit_signal = True
            elif position == -1 and close[i] >= camarilla_pp:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_4h_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_4h_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R4S4_Breakout_12hEMA50_Trend_VolumeConfirmation_PPExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0