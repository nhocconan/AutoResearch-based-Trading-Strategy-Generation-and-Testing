#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Camarilla R3/S3 breakout with 1d EMA50 trend filter, volume confirmation, and ATR trailing stop.
Long when price breaks above 1d Camarilla R3 level AND price > 1d EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below 1d Camarilla S3 level AND price < 1d EMA50 AND volume > 1.8x 20-period average.
Exit when price retraces to 1d Camarilla Pivot (midpoint) or ATR trailing stop hit (2.0*ATR from highest/lowest since entry).
Uses discrete position sizing (0.28) to balance return and drawdown.
Designed for 4h timeframe targeting ~30 trades/year per symbol (120 total over 4 years).
Combines strong structure (1d Camarilla pivot), trend (EMA), and momentum (volume) for robustness.
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
    
    # Calculate 1d Camarilla pivot levels (R3, S3, Pivot)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's high, low, close
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # Pivot = (High + Low + Close) / 3
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_r3 = c_1d + (h_1d - l_1d) * 1.1 / 4.0
    camarilla_s3 = c_1d - (h_1d - l_1d) * 1.1 / 4.0
    camarilla_pivot = (h_1d + l_1d + c_1d) / 3.0
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 1d EMA50 for trend filter
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume average (20-period) on 4h timeframe
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
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(1, 50, 20)  # Camarilla needs 1, EMA needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        pivot_val = camarilla_pivot_aligned[i]
        ema_50_val = ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1d Camarilla R3 AND price > 1d EMA50 AND volume spike
            if (price > r3_val and price > ema_50_val and volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.28
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below 1d Camarilla S3 AND price < 1d EMA50 AND volume spike
            elif (price < s3_val and price < ema_50_val and volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.28
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 1d Camarilla Pivot (midpoint)
            if position == 1 and price <= pivot_val:
                exit_signal = True
            elif position == -1 and price >= pivot_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.28 if position == 1 else -0.28
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0