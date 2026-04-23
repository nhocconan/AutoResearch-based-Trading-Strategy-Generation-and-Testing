#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
Long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) AND close > 1d EMA50 AND volume > 1.5x 20-period average.
Short when Alligator jaws < teeth < lips AND close < 1d EMA50 AND volume > 1.5x 20-period average.
Exit when Alligator lines cross in opposite direction (jaws < teeth for long, jaws > teeth for short) or ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) and volume filter to target 12-30 trades/year.
12h timeframe reduces noise and fee drag while capturing major trends in BTC/ETH across bull/bear regimes.
Williams Alligator identifies strong trending conditions proven effective in crypto markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA)"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=float)
    sma = pd.Series(source).rolling(window=period, min_periods=period).mean().values
    smma_val = np.full_like(source, np.nan, dtype=float)
    smma_val[period-1] = sma[period-1]
    for i in range(period, len(source)):
        smma_val[i] = (smma_val[i-1] * (period-1) + source[i]) / period
    return smma_val

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Williams Alligator (12h timeframe)
    jaws_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaws = smma(close, jaws_period)
    teeth = smma(close, teeth_period)
    lips = smma(close, lips_period)
    
    # ATR(14) for trailing stop calculation
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
    start_idx = max(50, jaws_period, teeth_period, lips_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_val = atr[i]
        ema50_val = ema50_1d_aligned[i]
        
        # Alligator conditions
        jaws_above_teeth = jaws[i] > teeth[i]
        teeth_above_lips = teeth[i] > lips[i]
        jaws_below_teeth = jaws[i] < teeth[i]
        teeth_below_lips = teeth[i] < lips[i]
        
        if position == 0:
            # Long: Alligator aligned up (jaws > teeth > lips) AND uptrend (price > EMA50) AND volume confirmation
            if jaws_above_teeth and teeth_above_lips and close[i] > ema50_val:
                # Volume confirmation: current volume > 1.5x 20-period average
                if i >= 20:
                    vol_ma = np.mean(volume[max(0, i-19):i+1])
                    if volume[i] > 1.5 * vol_ma:
                        signals[i] = 0.25
                        position = 1
                        highest_since_entry = price
            # Short: Alligator aligned down (jaws < teeth < lips) AND downtrend (price < EMA50) AND volume confirmation
            elif jaws_below_teeth and teeth_below_lips and close[i] < ema50_val:
                # Volume confirmation: current volume > 1.5x 20-period average
                if i >= 20:
                    vol_ma = np.mean(volume[max(0, i-19):i+1])
                    if volume[i] > 1.5 * vol_ma:
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
            
            # Primary exit: Alligator lines cross in opposite direction
            if position == 1 and (jaws[i] < teeth[i] or teeth[i] < lips[i]):
                exit_signal = True
            elif position == -1 and (jaws[i] > teeth[i] or teeth[i] > lips[i]):
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
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dEMA50_Trend_VolumeConfirmation_CrossExit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0