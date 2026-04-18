#!/usr/bin/env python3
"""
4h Bollinger Band Squeeze + Volume Spike + 1d RSI Filter
Hypothesis: Bollinger Band squeeze indicates low volatility, often preceding breakouts.
Volume spike confirms breakout strength. 1d RSI filter prevents counter-trend entries.
Works in bull (breakouts up when RSI>50) and bear (breakdowns down when RSI<50).
Designed for low trade frequency (<30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    basis = np.zeros(n)
    dev = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(n):
        if i < bb_length - 1:
            basis[i] = np.mean(close[0:i+1])
            dev[i] = np.std(close[0:i+1])
        else:
            basis[i] = np.mean(close[i-bb_length+1:i+1])
            dev[i] = np.std(close[i-bb_length+1:i+1])
        upper[i] = basis[i] + bb_mult * dev[i]
        lower[i] = basis[i] - bb_mult * dev[i]
    
    # Bollinger Band Width (normalized)
    bb_width = np.zeros(n)
    for i in range(n):
        if basis[i] != 0:
            bb_width[i] = (upper[i] - lower[i]) / basis[i]
        else:
            bb_width[i] = 0
    
    # Bollinger Band Squeeze: BB width below 20-period average
    bb_width_ma = np.zeros(n)
    for i in range(n):
        if i < 20:
            bb_width_ma[i] = np.mean(bb_width[0:i+1]) if i >= 0 else bb_width[i]
        else:
            bb_width_ma[i] = np.mean(bb_width[i-20+1:i+1])
    squeeze = bb_width < bb_width_ma
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Get daily data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI on daily closes
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 14:
            avg_gain[i] = np.mean(gain[0:i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[0:i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 0
    rsi_1d = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if rs[i] != 0:
            rsi_1d[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi_1d[i] = 50  # neutral when no loss
    
    # Align RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for BB and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(bb_width[i]) or np.isnan(bb_width_ma[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: BB squeeze breakout up with volume spike and bullish daily RSI (>50)
            if close[i] > upper[i] and vol_spike[i] and squeeze[i] and rsi_1d_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout down with volume spike and bearish daily RSI (<50)
            elif close[i] < lower[i] and vol_spike[i] and squeeze[i] and rsi_1d_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle band or volatility expands
            if close[i] < basis[i] or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle band or volatility expands
            if close[i] > basis[i] or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BB_Squeeze_VolumeSpike_RSIFilter"
timeframe = "4h"
leverage = 1.0