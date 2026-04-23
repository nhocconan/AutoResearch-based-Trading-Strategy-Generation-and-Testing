#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band squeeze breakout with 1w trend filter and volume confirmation.
Long when price breaks above upper BB(20,2) AND 1w close > 1w EMA50 AND volume > 2x 20-period average.
Short when price breaks below lower BB(20,2) AND 1w close < 1w EMA50 AND volume > 2x 20-period average.
Exit when price returns to middle BB(20) or ATR-based stoploss (2.0x ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-25 trades/year per symbol.
Bollinger Band squeeze captures low volatility breakouts, while 1w EMA50 ensures alignment with weekly trend.
Volume confirmation filters weak breakouts. Works in both bull and bear markets by trading with the weekly trend.
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
    
    # Load 6h data for Bollinger Bands and ATR calculation - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Bollinger Bands (20,2) on 6h data
    sma_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    middle_bb = sma_20
    
    # Align 6h Bollinger Bands to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_6h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_6h, lower_bb)
    middle_bb_aligned = align_htf_to_ltf(prices, df_6h, middle_bb)
    
    # Calculate ATR(14) on 6h data for stoploss
    tr1 = np.maximum(high_6h - low_6h, np.abs(high_6h - np.roll(close_6h, 1)))
    tr2 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_6h[0] - low_6h[0]  # first bar
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(middle_bb_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above upper BB AND 1w close > 1w EMA50 AND volume spike
            if (price > upper_bb_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below lower BB AND 1w close < 1w EMA50 AND volume spike
            elif (price < lower_bb_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to middle BB or ATR stoploss
                if price < middle_bb_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.0 * atr_6h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to middle BB or ATR stoploss
                if price > middle_bb_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.0 * atr_6h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Bollinger_Squeeze_Breakout_1wEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0