#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1d weekly Camarilla R4/S4 breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above 1d weekly Camarilla R4 level AND price > 1w EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below 1d weekly Camarilla S4 level AND price < 1w EMA34 AND volume > 2.0x 20-period average.
Exit when price retraces to 1d weekly Camarilla Pivot (midpoint) or ATR(14) trailing stop hit (2.5*ATR from highest/lowest since entry).
Uses discrete position sizing (0.25) to control drawdown and minimize fee churn.
Designed for 6h timeframe targeting ~25 trades/year per symbol (100 total over 4 years).
Combines weekly structure (Camarilla R4/S4 from daily data), trend (1w EMA), and momentum (volume) for robustness in both bull and bear markets.
Weekly Camarilla levels derived from 1d data aggregated to weekly: uses weekly high, low, close.
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
    
    # Calculate 1d data then resample to weekly for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Resample 1d to weekly using actual Binance weekly boundaries (via get_htf_data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly Camarilla levels: based on previous week's high, low, close
    # R4 = Close + (High - Low) * 1.1/2
    # S4 = Close - (High - Low) * 1.1/2
    # Pivot = (High + Low + Close) / 3
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    camarilla_r4 = c_1w + (h_1w - l_1w) * 1.1 / 2.0
    camarilla_s4 = c_1w - (h_1w - l_1w) * 1.1 / 2.0
    camarilla_pivot = (h_1w + l_1w + c_1w) / 3.0
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Calculate 1w EMA34 for trend filter
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume average (20-period) on 6h timeframe
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
    start_idx = max(1, 34, 20)  # Weekly Camarilla needs 1, EMA needs 34, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        pivot_val = camarilla_pivot_aligned[i]
        ema_34_val = ema_34_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1w Camarilla R4 AND price > 1w EMA34 AND volume spike
            if (price > r4_val and price > ema_34_val and volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below 1w Camarilla S4 AND price < 1w EMA34 AND volume spike
            elif (price < s4_val and price < ema_34_val and volume[i] > 2.0 * vol_ma_val):
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
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
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

name = "6H_WeeklyCamarilla_R4S4_Breakout_1wEMA34_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0