#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Camarilla pivot (R1/S1) breakout with volume confirmation and 1d EMA34 trend filter.
Long when price breaks above 4h R1 AND volume > 1.3x 20-period average AND close > 1d EMA34.
Short when price breaks below 4h S1 AND volume > 1.3x 20-period average AND close < 1d EMA34.
Exit when price retraces to 4h pivot point (PP) or ATR trailing stop hit (2.0*ATR from highest/lowest since entry).
Uses discrete position sizing (0.20) to control drawdown and fee churn.
Designed for 1h timeframe to target 15-37 trades/year per symbol (60-150 total over 4 years).
Camarilla pivots provide precise intraday support/resistance levels. Volume confirmation filters false breakouts.
1d EMA34 ensures we only trade with the higher timeframe trend. Works in both bull and bear markets by
capturing strong directional moves while avoiding choppy periods through volume and trend filters.
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
    
    # Calculate 4h Camarilla pivots (R1, S1, PP)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla calculations for each 4h bar
    camarilla_pp = (high_4h + low_4h + close_4h) / 3.0
    camarilla_r1 = close_4h + (high_4h - low_4h) * 1.1 / 12.0
    camarilla_s1 = close_4h - (high_4h - low_4h) * 1.1 / 12.0
    
    # Align Camarilla levels to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (20-period) on 1h timeframe
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
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        pp = camarilla_pp_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema_34 = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 4h R1 AND volume spike AND above 1d EMA34 (uptrend)
            if (price > r1 and volume[i] > 1.3 * vol_ma_val and price > ema_34):
                signals[i] = 0.20
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below 4h S1 AND volume spike AND below 1d EMA34 (downtrend)
            elif (price < s1 and volume[i] > 1.3 * vol_ma_val and price < ema_34):
                signals[i] = -0.20
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
            
            # Primary exit: Price retraces to 4h Camarilla pivot point (PP)
            if position == 1 and price <= pp:
                exit_signal = True
            elif position == -1 and price >= pp:
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
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_VolumeConfirmation_1dEMA34Trend_ATRTrailingStop"
timeframe = "1h"
leverage = 1.0