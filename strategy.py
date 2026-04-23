#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA200 trend filter and volume confirmation.
Long when Bear Power < 0 (bulls in control) AND close > 1d EMA200 AND volume > 1.5x 20-period average.
Short when Bull Power > 0 (bears in control) AND close < 1d EMA200 AND volume > 1.5x 20-period average.
Exit when Elder Power reverses sign or ATR-based stoploss hits.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-30 trades/year per symbol.
6h timeframe reduces noise while 1d EMA200 provides robust trend filter that works in both bull and bear markets.
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
    
    # Load 6h data for Elder Ray calculation - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 13-period EMA for Elder Ray (standard period)
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power_6h = high_6h - ema13_6h  # Bull Power = High - EMA13
    bear_power_6h = low_6h - ema13_6h   # Bear Power = Low - EMA13
    
    # Align 6h Elder Ray to 6h timeframe (no additional delay needed)
    bull_power_6h_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_6h_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    
    # Load 1d data for EMA200 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d data
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 6h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 6h data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bull_power_6h_aligned[i]) or np.isnan(bear_power_6h_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bear Power < 0 (bulls in control) AND close > 1d EMA200 AND volume spike
            if (bear_power_6h_aligned[i] < 0 and 
                close[i] > ema200_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Bull Power > 0 (bears in control) AND close < 1d EMA200 AND volume spike
            elif (bull_power_6h_aligned[i] > 0 and 
                  close[i] < ema200_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Bear Power becomes positive OR ATR stoploss
                if bear_power_6h_aligned[i] >= 0:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr_6h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bull Power becomes negative OR ATR stoploss
                if bull_power_6h_aligned[i] <= 0:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr_6h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1dEMA200_VolumeConfirm"
timeframe = "6h"
leverage = 1.0