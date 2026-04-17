#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout + volume spike + 4h EMA50 trend filter + ATR trailing stop
- Camarilla H3/L3 breakouts on 1h capture intraday momentum with controlled frequency
- Volume spike (>2.0x 20-period average) confirms breakout strength and reduces false signals
- 4h EMA50 filter ensures alignment with higher timeframe trend to avoid counter-trend trades
- ATR-based trailing stop (2.5 * ATR from extreme) manages risk while allowing trends to run
- Position sizing: 0.20 discrete to minimize fee churn
- Session filter: 08-20 UTC to avoid low-volume Asian session noise
- Target: 15-35 trades/year per symbol (~60-140 total over 4 years)
- Works in bull markets (captures breakouts) and bear markets (short breakdowns with trend filter)
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
    
    # Get 1h data for primary timeframe calculations
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivots on 1h (based on previous bar's OHLC)
    def calculate_camarilla(high_arr, low_arr, close_arr):
        # Camarilla levels based on previous period
        pivot = (high_arr + low_arr + close_arr) / 3.0
        range_val = high_arr - low_arr
        H3 = pivot + range_val * 1.1 / 4
        L3 = pivot - range_val * 1.1 / 4
        return H3, L3
    
    # Shift by 1 to use previous bar's levels (no look-ahead)
    camarilla_H3, camarilla_L3 = calculate_camarilla(high_1h, low_1h, close_1h)
    camarilla_H3 = np.roll(camarilla_H3, 1)
    camarilla_L3 = np.roll(camarilla_L3, 1)
    camarilla_H3[0] = np.nan
    camarilla_L3[0] = np.nan
    
    # Calculate EMA50 on 4h for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 1h
    volume_ma_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss on 1h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr_1h = calculate_atr(high_1h, low_1h, close_1h, 14)
    
    # Align all indicators to 1h timeframe
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_L3)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1h, volume_ma_1h)
    atr_aligned = align_htf_to_ltf(prices, df_1h, atr_1h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    long_high = 0.0   # track highest close since entry for trailing stop
    low_low = 0.0     # track lowest close since entry for trailing stop
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        H3 = camarilla_H3_aligned[i]
        L3 = camarilla_L3_aligned[i]
        ema_trend = ema50_4h_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above H3 + volume spike + price > 4h EMA50 (uptrend)
            if price > H3 and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.20
                position = 1
                entry_price = price
                long_high = price
            # Short: price breaks below L3 + volume spike + price < 4h EMA50 (downtrend)
            elif price < L3 and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.20
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
                signals[i] = 0.20
        
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
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_VolumeSpike_ATRTrail"
timeframe = "1h"
leverage = 1.0