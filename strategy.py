#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1-week Camarilla pivot R4/S4 levels breakout with volume confirmation and 1-day EMA50 trend filter.
Long when price breaks above 1w Camarilla R4 level AND volume > 1.8x 20-period average AND close > 1d EMA50.
Short when price breaks below 1w Camarilla S4 level AND volume > 1.8x 20-period average AND close < 1d EMA50.
Exit when price retraces to the 1w Camarilla midpoint (R4-S4/2) or ATR trailing stop hit (2.0*ATR from highest/lowest since entry).
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 6h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Weekly Camarilla pivots identify key support/resistance levels derived from prior week's range.
Breakouts above R4 or below S4 with volume confirmation and trend alignment indicate strong institutional interest with momentum.
Works in both bull and bear markets by capturing strong directional moves while avoiding false breakouts in ranging markets.
"""

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
    
    # Calculate 1-week Camarilla pivot levels (R4, S4, midpoint)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Camarilla levels based on previous week's OHLC
    # R4 = Close + 1.1 * 2 * (High - Low) = Close + 2.2 * (High - Low)
    # S4 = Close - 1.1 * 2 * (High - Low) = Close - 2.2 * (High - Low)
    # Midpoint = (R4 + S4)/2 = Close
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    r4 = prev_close + 2.2 * (prev_high - prev_low)
    s4 = prev_close - 2.2 * (prev_high - prev_low)
    midpoint = prev_close  # (R4 + S4)/2 simplifies to previous close
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    midpoint_aligned = align_htf_to_ltf(prices, df_1w, midpoint)
    
    # Calculate 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
    start_idx = max(20, 50, 2)  # volume MA needs 20, EMA50 needs 50, 1w data needs at least 2 for shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        mid_val = midpoint_aligned[i]
        ema_50_val = ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1w Camarilla R4 level AND volume spike AND above 1d EMA50
            if (price > r4_val and volume[i] > 1.8 * vol_ma_val and close[i] > ema_50_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below 1w Camarilla S4 level AND volume spike AND below 1d EMA50
            elif (price < s4_val and volume[i] > 1.8 * vol_ma_val and close[i] < ema_50_val):
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
            
            # Primary exit: Price retraces to 1w Camarilla midpoint (previous week close)
            if position == 1 and price <= mid_val:
                exit_signal = True
            elif position == -1 and price >= mid_val:
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

name = "6H_WeeklyCamarilla_R4S4_Breakout_VolumeConfirmation_EMA50Trend_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0