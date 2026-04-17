#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout + volume spike + 1d EMA50 trend filter + ATR trailing stop
- Camarilla H3/L3 levels from 1d provide institutional pivot points with high probability reactions
- Volume spike (>2.0x 20-period average) confirms breakout validity and reduces false signals
- 1d EMA50 filter ensures alignment with daily trend to avoid counter-trend trades
- ATR-based trailing stop (2.5 * ATR from extreme) manages risk while allowing trends to develop
- Position sizing: 0.25 discrete to minimize fee churn
- Target: 20-40 trades/year per symbol (~80-160 total over 4 years)
- Works in bull markets (captures breakouts above H3 in uptrend) and bear markets (short breakdowns below L3 in downtrend)
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
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for primary timeframe calculations
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels (H3, L3) from 1d OHLC
    def calculate_camarilla(high_arr, low_arr, close_arr):
        # Camarilla pivot levels
        pivot = (high_arr + low_arr + close_arr) / 3.0
        range_val = high_arr - low_arr
        H3 = close_arr + range_val * 1.1 / 4.0
        L3 = close_arr - range_val * 1.1 / 4.0
        return H3, L3
    
    H3_1d, L3_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Volume average (20-period) on 4h
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss on 4h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, 14)
    
    # Align all indicators to 4h timeframe
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    long_high = 0.0   # track highest close since entry for trailing stop
    low_low = 0.0     # track lowest close since entry for trailing stop
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        H3 = H3_1d_aligned[i]
        L3 = L3_1d_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above H3 + volume spike + price > 1d EMA50 (uptrend)
            if price > H3 and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
                long_high = price
            # Short: price breaks below L3 + volume spike + price < 1d EMA50 (downtrend)
            elif price < L3 and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
                low_low = price
        
        elif position == 1:
            # Update highest close since entry
            if price > long_high:
                long_high = price
            
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: ATR trailing stop (2.5 * ATR from highest close)
            if price < long_high - 2.5 * atr_val:
                exit_signal = True
            
            # Exit 2: Price retrace to midpoint between H3 and L3
            elif price < (H3 + L3) / 2:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest close since entry
            if price < low_low:
                low_low = price
            
            # Exit conditions for short
            exit_signal = False
            
            # Exit 1: ATR trailing stop (2.5 * ATR from lowest close)
            if price > low_low + 2.5 * atr_val:
                exit_signal = True
            
            # Exit 2: Price retrace to midpoint between H3 and L3
            elif price > (H3 + L3) / 2:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA50_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0