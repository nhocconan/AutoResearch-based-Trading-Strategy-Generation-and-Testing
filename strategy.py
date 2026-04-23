#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA50 trend filter and ATR volatility filter.
Elder Ray measures bull/bear power relative to EMA13 to identify trend strength.
Only take trades in direction of 1d EMA50 trend when volatility is normal (ATR ratio < 1.5).
Avoids whipsaws in ranging markets and captures sustained moves in both bull and bear regimes.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Calculate 6h EMA13 for Elder Ray
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_6h - ema_13_6h
    bear_power = low_6h - ema_13_6h
    
    # Align Elder Ray to 6h timeframe (previous 6h bar values)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # Calculate 1d EMA50 for primary trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volatility filter on 6h
    atr_period = 14
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14_6h = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_14_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_14_6h)
    
    # ATR ratio: current ATR / 50-period MA of ATR (to detect abnormal volatility)
    atr_ma_50 = pd.Series(atr_14_6h_aligned).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14_6h_aligned / atr_ma_50
    normal_volatility = atr_ratio < 1.5  # avoid high volatility whipsaws
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13)  # need EMA50 and EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND uptrend on 1d AND normal volatility
            if bull_power_aligned[i] > 0 and trend_up and normal_volatility[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND downtrend on 1d AND normal volatility
            elif bear_power_aligned[i] < 0 and trend_down and normal_volatility[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: opposite Elder Ray signal OR volatility expansion
            exit_signal = False
            if position == 1:
                # Exit long on Bear Power >= 0 (loss of bullish momentum)
                if bear_power_aligned[i] >= 0:
                    exit_signal = True
            elif position == -1:
                # Exit short on Bull Power <= 0 (loss of bearish momentum)
                if bull_power_aligned[i] <= 0:
                    exit_signal = True
            
            # Also exit on volatility expansion ( ATR ratio > 2.0 )
            if atr_ratio[i] > 2.0:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1dEMA50_Trend_VolFilter"
timeframe = "6h"
leverage = 1.0