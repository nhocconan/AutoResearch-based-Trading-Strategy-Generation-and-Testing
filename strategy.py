#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R4 (1d) AND close > 1w EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S4 (1d) AND close < 1w EMA50 AND volume > 2.0x 20-period average.
Exit when price retraces to Camarilla pivot point (PP) OR ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~15-25 trades/year on 12h timeframe.
Camarilla R4/S4 provides extreme breakout levels for high-quality signals in both bull and bear markets.
1w EMA50 filter ensures alignment with weekly trend, reducing whipsaws.
Volume confirmation adds conviction to breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Camarilla levels from 1d (PP, R4, S4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot point
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla R4 and S4 (more extreme than R3/S3)
    r4 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    s4 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
    start_idx = max(50, 20, 14)  # EMA50 needs 50, vol MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_1w_aligned[i]
        pp_val = pp_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R4 AND uptrend (price > EMA50) AND volume spike (2.0x avg)
            if close[i] > r4_val and close[i] > ema50_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S4 AND downtrend (price < EMA50) AND volume spike (2.0x avg)
            elif close[i] < s4_val and close[i] < ema50_val and volume[i] > 2.0 * vol_ma_val:
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
            
            # Primary exit: Price retraces to Camarilla pivot point (PP)
            if position == 1 and close[i] <= pp_val:
                exit_signal = True
            elif position == -1 and close[i] >= pp_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R4S4_Breakout_1wEMA50_Trend_VolumeConfirmation_PPExit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0