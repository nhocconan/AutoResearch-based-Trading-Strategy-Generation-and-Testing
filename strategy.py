#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using weekly Williams Alligator trend filter with daily RSI reversal signals.
In strong weekly trends (Alligator aligned), daily RSI extremes (<30 or >70) often precede mean-reversion moves.
Volume confirmation (>1.5x average) filters weak signals. Uses discrete position sizing (0.0, ±0.25) to minimize fee churn.
Target: 10-25 trades/year (40-100 over 4 years). Includes ATR-based stoploss for risk control.
Works in both bull (buy RSI<30 in uptrend) and bear (sell RSI>70 in downtrend) markets.
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
    
    # Get weekly data for Alligator (Jaw, Teeth, Lips)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate Alligator components (13,8,5 SMAs shifted)
    close_1w = df_1w['close'].values
    jaw = np.full(len(close_1w), np.nan)
    teeth = np.full(len(close_1w), np.nan)
    lips = np.full(len(close_1w), np.nan)
    
    # Jaw: 13-period SMA shifted 8 bars
    if len(close_1w) >= 13:
        for i in range(12, len(close_1w)):
            jaw[i] = np.mean(close_1w[i-12:i+1])
    jaw_shifted = np.full_like(jaw, np.nan)
    if len(jaw) >= 8:
        jaw_shifted[8:] = jaw[:-8]
    
    # Teeth: 8-period SMA shifted 5 bars
    if len(close_1w) >= 8:
        for i in range(7, len(close_1w)):
            teeth[i] = np.mean(close_1w[i-7:i+1])
    teeth_shifted = np.full_like(teeth, np.nan)
    if len(teeth) >= 5:
        teeth_shifted[5:] = teeth[:-5]
    
    # Lips: 5-period SMA shifted 3 bars
    if len(close_1w) >= 5:
        for i in range(4, len(close_1w)):
            lips[i] = np.mean(close_1w[i-4:i+1])
    lips_shifted = np.full_like(lips, np.nan)
    if len(lips) >= 3:
        lips_shifted[3:] = lips[:-3]
    
    # Align weekly Alligator to daily (waits for weekly bar close)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_shifted)
    
    # Daily RSI (14-period)
    rsi_period = 14
    rsi = np.full(n, np.nan)
    if n >= rsi_period + 1:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        if n >= rsi_period + 1:
            avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
            avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
            
            for i in range(rsi_period + 1, n):
                avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
                avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
                
                rs = np.where(avg_loss[i] != 0, avg_gain[i] / avg_loss[i], 0)
                rsi[i] = 100 - (100 / (1 + rs))
    
    # Daily average volume for spike detection
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # ATR for stoploss
    atr_period = 14
    tr = np.zeros(n)
    atr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(atr_period, n):
        if i == atr_period:
            atr[i] = np.mean(tr[1:atr_period+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need weekly indicators + RSI + volume + ATR
    start_idx = max(13+8, rsi_period+1, vol_period, atr_period)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine Alligator alignment (trend direction)
        # Bullish: Lips > Teeth > Jaw
        # Bearish: Lips < Teeth < Jaw
        bullish_align = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_align = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long signal: RSI < 30 (oversold) in bullish Alligator alignment with volume
            if bullish_align and rsi[i] < 30 and volume_confirmation:
                signals[i] = size
                position = 1
            # Short signal: RSI > 70 (overbought) in bearish Alligator alignment with volume
            elif bearish_align and rsi[i] > 70 and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 50 (neutral) or Alligator turns bearish or stoploss hit
            if rsi[i] > 50 or not bullish_align or price < (entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RSI < 50 (neutral) or Alligator turns bullish or stoploss hit
            if rsi[i] < 50 or not bearish_align or price > (entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
        
        # Track entry price for stoploss calculation
        if position != 0 and signals[i] != 0:
            if position == 1 and signals[i] == size:
                entry_price = price
            elif position == -1 and signals[i] == -size:
                entry_price = price
    
    return signals

name = "1d_Alligator_RSI_Volume"
timeframe = "1d"
leverage = 1.0