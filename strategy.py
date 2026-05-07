#!/usr/bin/env python3
name = "6h_1d_Trix_Trend_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate TRIX on daily close (15-period EMA triple)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # Signal line: 9-period EMA of TRIX
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX and signal to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_raw)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    
    # Daily trend filter: EMA(50) on daily close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 4, 15)  # Wait for EMA, volume MA, and TRIX
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line with volume and daily uptrend
            trix_cross_up = trix_aligned[i] > trix_signal_aligned[i] and trix_aligned[i-1] <= trix_signal_aligned[i-1]
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if trix_cross_up and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line with volume and daily downtrend
            elif trix_aligned[i] < trix_signal_aligned[i] and trix_aligned[i-1] >= trix_signal_aligned[i-1] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below signal line or volume drops
            trix_cross_down = trix_aligned[i] < trix_signal_aligned[i] and trix_aligned[i-1] >= trix_signal_aligned[i-1]
            if trix_cross_down or volume[i] < vol_ma_4[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above signal line or volume drops
            trix_cross_up = trix_aligned[i] > trix_signal_aligned[i] and trix_aligned[i-1] <= trix_signal_aligned[i-1]
            if trix_cross_up or volume[i] < vol_ma_4[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h TRIX trend reversal with daily trend filter and volume confirmation
# - TRIX (triple EMA) identifies momentum changes and overbought/oversold conditions
# - Long when TRIX crosses above signal line with volume spike in daily uptrend
# - Short when TRIX crosses below signal line with volume spike in daily downtrend
# - Daily EMA(50) filter ensures trades align with higher timeframe trend
# - Volume confirmation (1.5x average) filters false signals
# - Exit on TRIX signal reversal or volume drying up
# - Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# - Targets 50-100 total trades over 4 years (12-25/year) to minimize fee drag
# - Position size 0.25 balances opportunity and risk control