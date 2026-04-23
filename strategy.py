#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND 4h close > 4h EMA20 AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S3 AND 4h close < 4h EMA20 AND volume > 1.5x 20-period average.
Exit when price retraces to Camarilla pivot point (PP) or ATR trailing stop (1.5*ATR from extreme).
Uses discrete position sizing (0.20) to minimize fee drag.
Target trade frequency: 15-37/year per symbol (60-150 total over 4 years) to avoid fee drag on 1h timeframe.
Uses 4h EMA20 for trend filter and 1d OHLC for Camarilla levels to ensure alignment with higher timeframe structure.
Works in bull markets (breakouts with volume in uptrend) and bear markets (breakdowns with volume in downtrend).
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
    
    # Calculate 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate Camarilla levels from 1d OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculation: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_pp_1d = typical_price_1d
    camarilla_r3_1d = close_1d + (range_1d * 1.1 / 4)
    camarilla_s3_1d = close_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # EMA20 needs 20, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema20_val = ema20_4h_aligned[i]
        camarilla_pp = camarilla_pp_aligned[i]
        camarilla_r3 = camarilla_r3_aligned[i]
        camarilla_s3 = camarilla_s3_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND uptrend (4h close > EMA20) AND volume spike
            if close[i] > camarilla_r3 and close[i] > ema20_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.20
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S3 AND downtrend (4h close < EMA20) AND volume spike
            elif close[i] < camarilla_s3 and close[i] < ema20_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.20
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
            
            # ATR-based trailing stop: 1.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 1.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 1.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_4hEMA20_Trend_VolumeConfirmation_PPExit_ATRTrailingStop"
timeframe = "1h"
leverage = 1.0