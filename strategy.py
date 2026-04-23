#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1w Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume confirmation, and ATR trailing stop.
Long when price breaks above 1w Camarilla R3 level AND price > 1d EMA34 AND volume > 1.8x 24-period average.
Short when price breaks below 1w Camarilla S3 level AND price < 1d EMA34 AND volume > 1.8x 24-period average.
Exit when price retraces to 1w Camarilla Pivot (midpoint) or ATR trailing stop hit (2.0*ATR from highest/lowest since entry).
Uses discrete position sizing (0.25) to balance return and drawdown.
Designed for 12h timeframe targeting ~20-30 trades/year per symbol (80-120 total over 4 years).
Combines strong structure (1w Camarilla pivot), trend (EMA), and momentum (volume) for robustness.
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
    
    # Calculate 1w Camarilla pivot levels (R3, S3, Pivot)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous week's high, low, close
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # Pivot = (High + Low + Close) / 3
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    camarilla_r3 = c_1w + (h_1w - l_1w) * 1.1 / 4.0
    camarilla_s3 = c_1w - (h_1w - l_1w) * 1.1 / 4.0
    camarilla_pivot = (h_1w + l_1w + c_1w) / 3.0
    
    # Align 1w Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume average (24-period = 12d) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
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
    start_idx = max(1, 34, 24)  # Camarilla needs 1, EMA needs 34, vol MA needs 24
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
        ema_34_val = ema_34_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1w Camarilla R3 AND price > 1d EMA34 AND volume spike
            if (price > r3_val and price > ema_34_val and volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below 1w Camarilla S3 AND price < 1d EMA34 AND volume spike
            elif (price < s3_val and price < ema_34_val and volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
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
            
            # Primary exit: Price retraces to 1w Camarilla Pivot (midpoint)
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
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0