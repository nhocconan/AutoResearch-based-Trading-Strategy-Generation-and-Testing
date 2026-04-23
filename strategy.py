#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND close > 4h EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S1 AND close < 4h EMA50 AND volume > 2.0x 20-period average.
Exit when price retraces to Camarilla pivot point (PP) OR ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.20) targeting 15-37 trades/year on 1h timeframe.
Adds session filter (08-20 UTC) to reduce noise trades.
Camarilla R1/S1 breakouts with volume and trend alignment capture momentum moves with fewer false signals.
Session filter avoids low-liquidity periods that increase whipsaw.
Works in bull (trend-following breakouts) and bear (mean-reversion at extreme levels via volatility expansion).
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
    open_time = prices['open_time'].values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla pivot levels from 1d (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: PP = (H+L+C)/3, R1 = PP + (H-L)*1.1/12, S1 = PP - (H-L)*1.1/12
    pivot = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r1 = pivot + (high_1d - low_1d) * 1.1 / 12.0
    camarilla_s1 = pivot - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
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
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50 needs 50, vol MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_4h_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        pivot_val = pivot_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R1 AND uptrend (price > EMA50) AND volume spike (2.0x avg)
            if close[i] > r1_val and close[i] > ema50_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.20
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S1 AND downtrend (price < EMA50) AND volume spike (2.0x avg)
            elif close[i] < s1_val and close[i] < ema50_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.20
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
            
            # Primary exit: Price retraces to Camarilla pivot point
            if position == 1 and close[i] <= pivot_val:
                exit_signal = True
            elif position == -1 and close[i] >= pivot_val:
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
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeConfirmation_PivotExit_ATRTrailingStop_SessionFilter"
timeframe = "1h"
leverage = 1.0