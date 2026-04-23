#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above 6h Camarilla R3 AND 1d close > 1d EMA34 AND 6h volume > 2.0x 20-period average volume.
Short when price breaks below 6h Camarilla S3 AND 1d close < 1d EMA34 AND 6h volume > 2.0x 20-period average volume.
Exit when price reaches the 6h Camarilla pivot point (PP) OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~12-37 trades/year on 6h timeframe.
Camarilla levels provide intraday structure that adapts to volatility, effective in both trending and ranging markets.
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
    
    # Calculate 6h OHLC for Camarilla
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need enough for calculations
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Previous 6h Camarilla levels (R3, S3, PP)
    camarilla_pp = (high_6h + low_6h + close_6h) / 3.0
    camarilla_range = high_6h - low_6h
    camarilla_r3 = camarilla_pp + camarilla_range * 1.1 / 4.0
    camarilla_s3 = camarilla_pp - camarilla_range * 1.1 / 4.0
    
    # Align Camarilla levels to lower timeframe (15m)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_6h, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 6h trailing stop calculation
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
    start_idx = max(20, 34)  # Volume MA20 and EMA34 need 20 and 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema_val = ema_34_1d_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        pp = camarilla_pp_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND bullish trend (close > EMA34) AND volume spike
            if close[i] > r3 and close_1d[-1] > ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S3 AND bearish trend (close < EMA34) AND volume spike
            elif close[i] < s3 and close_1d[-1] < ema_val and volume[i] > 2.0 * vol_ma_val:
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
            
            # Primary exit: Price reaches Camarilla pivot point (PP)
            if position == 1 and close[i] >= pp:
                exit_signal = True
            elif position == -1 and close[i] <= pp:
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

name = "6H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation_PPExit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0