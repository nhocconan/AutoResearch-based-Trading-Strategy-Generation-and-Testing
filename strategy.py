#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R1 AND 4h EMA50 uptrend AND volume > 1.8x 20-period average.
Short when price breaks below Camarilla S1 AND 4h EMA50 downtrend AND volume > 1.8x 20-period average.
Exit when price retouches Camarilla pivot point (PP) or ATR stoploss hit (1.5*ATR).
Uses discrete position sizing (0.20) to balance return and risk. Targets 15-37 trades/year per symbol.
Camarilla levels provide structure, 4h EMA50 ensures alignment with intermediate trend, volume filters weak breakouts.
Designed for 1h timeframe to reduce trade frequency and fee drag while capturing intermediate swings.
Works in both bull (trend continuation) and bear (mean reversion at extremes) markets.
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
    
    # Calculate Camarilla levels from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day's OHLC)
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r1 = camarilla_pp + (high_1d - low_1d) * 1.1 / 12.0
    camarilla_s1 = camarilla_pp - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Load 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume average (20-period) on 1h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation (using 1h data)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
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
        ema50 = ema50_4h_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND 4h EMA50 uptrend AND volume spike
            if (price > r1 and 
                close[i] > ema50 and  # Current close above EMA50 for uptrend
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: Price breaks below Camarilla S1 AND 4h EMA50 downtrend AND volume spike
            elif (price < s1 and 
                  close[i] < ema50 and  # Current close below EMA50 for downtrend
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Camarilla pivot point
            if position == 1 and price <= pp:
                exit_signal = True
            elif position == -1 and price >= pp:
                exit_signal = True
            
            # ATR-based stoploss: 1.5 * ATR from entry (tighter for 1h)
            if position == 1 and price < entry_price - 1.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 1.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_4hEMA50_VolumeSpike_ATRStop"
timeframe = "1h"
leverage = 1.0