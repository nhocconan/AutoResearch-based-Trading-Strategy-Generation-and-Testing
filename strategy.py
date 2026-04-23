#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1-week Camarilla R4/S4 breakouts with 1-day EMA50 trend filter and volume confirmation.
Long when price breaks above weekly R4 AND close > 1d EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below weekly S4 AND close < 1d EMA50 AND volume > 1.8x 20-period average.
Exit when price retraces to weekly pivot point (PP) or ATR trailing stop (2.0*ATR from extreme).
Weekly Camarilla provides strong structural levels; EMA50 filters trend; volume confirms breakout strength.
Designed for low trade frequency (12-37/year) to minimize fee drag on 6h timeframe.
Works in bull markets (strong breakouts with volume) and bear markets (breakdowns with volume).
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1w Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    PP_1w = (high_1w + low_1w + close_1w) / 3.0
    R4_1w = PP_1w + (high_1w - low_1w) * 1.1
    S4_1w = PP_1w - (high_1w - low_1w) * 1.1
    
    # Align 1w Camarilla levels to 6h timeframe
    PP_1w_aligned = align_htf_to_ltf(prices, df_1w, PP_1w)
    R4_1w_aligned = align_htf_to_ltf(prices, df_1w, R4_1w)
    S4_1w_aligned = align_htf_to_ltf(prices, df_1w, S4_1w)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 2)  # EMA50 needs 50, vol MA needs 20, Camarilla needs 2
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(R4_1w_aligned[i]) or np.isnan(S4_1w_aligned[i]) or 
            np.isnan(PP_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_1d_aligned[i]
        R4_val = R4_1w_aligned[i]
        S4_val = S4_1w_aligned[i]
        PP_val = PP_1w_aligned[i]
        
        if position == 0:
            # Long: Break above weekly R4 AND uptrend (close > EMA50) AND volume spike
            if close[i] > R4_val and close[i] > ema50_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below weekly S4 AND downtrend (close < EMA50) AND volume spike
            elif close[i] < S4_val and close[i] < ema50_val and volume[i] > 1.8 * vol_ma_val:
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
            
            # Primary exit: Price retraces to weekly pivot point (PP)
            if position == 1 and close[i] <= PP_val:
                exit_signal = True
            elif position == -1 and close[i] >= PP_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WeeklyCamarilla_R4S4_Breakout_1dEMA50_Trend_VolumeSpike_ATRTrailingStop_PPExit"
timeframe = "6h"
leverage = 1.0