#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND close > 1d EMA50 AND volume > 1.5x 24-period average.
Short when price breaks below Camarilla S3 AND close < 1d EMA50 AND volume > 1.5x 24-period average.
Exit when price reaches Camarilla PP (pivot point) OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) to target 12-37 trades/year on 12h timeframe.
Camarilla levels from daily timeframe provide institutional structure proven in ranging and trending markets.
12h timeframe reduces noise and overtrading while capturing multi-day moves in BTC/ETH.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # We need the previous completed 1d bar's OHLC
    df_1d_prev = df_1d.copy()
    # Shift to get previous day's values (avoid look-ahead)
    high_1d_prev = df_1d_prev['high'].shift(1).values
    low_1d_prev = df_1d_prev['low'].shift(1).values
    close_1d_prev = df_1d_prev['close'].shift(1).values
    
    # Align to LTF (12h) - each value represents the PREVIOUS day's Camarilla levels
    # Available only after the 1d bar closes
    high_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, high_1d_prev)
    low_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, low_1d_prev)
    close_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, close_1d_prev)
    
    # Calculate Camarilla levels for previous day
    range_1d = high_1d_prev_aligned - low_1d_prev_aligned
    camarilla_pp = (high_1d_prev_aligned + low_1d_prev_aligned + close_1d_prev_aligned) / 3.0
    camarilla_r3 = camarilla_pp + (range_1d * 1.1 / 4.0)
    camarilla_s3 = camarilla_pp - (range_1d * 1.1 / 4.0)
    
    # Volume average (24-period = 12 days of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR(20) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24, 20)  # EMA50 needs 50, vol MA needs 24, ATR needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(camarilla_pp[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_1d_aligned[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        pp = camarilla_pp[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND uptrend (price > EMA50) AND volume spike (1.5x avg)
            if close[i] > r3 and close[i] > ema50_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S3 AND downtrend (price < EMA50) AND volume spike (1.5x avg)
            elif close[i] < s3 and close[i] < ema50_val and volume[i] > 1.5 * vol_ma_val:
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
            
            # Primary exit: Price reaches Camarilla PP (pivot point)
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

name = "12H_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeConfirmation_PPExit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0