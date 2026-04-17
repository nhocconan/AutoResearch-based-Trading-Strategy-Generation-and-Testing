#!/usr/bin/env python3
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
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Shift to use previous day's pivots (avoid look-ahead)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    
    # Align daily pivot levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_prev)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_prev)
    
    # Volume confirmation: current volume > 1.5 * 48-period average (4h * 12 = 48h)
    volume_ma48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Momentum filter: 4-period RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=4, min_periods=4).mean().values
    avg_loss = pd.Series(loss).rolling(window=4, min_periods=4).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 50.0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 48  # Need volume MA48 and ATR MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma48[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma20[i]) or 
            np.isnan(r1_4h[i]) or 
            np.isnan(s1_4h[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 48-period average
        volume_filter = volume[i] > (1.5 * volume_ma48[i])
        # Volatility filter: ATR > ATR MA20 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma20[i]
        # Momentum filter: RSI between 30 and 70 (avoid extremes)
        momentum_filter = (rsi[i] >= 30) & (rsi[i] <= 70)
        
        if position == 0:
            # Long: price breaks above R1 with volume, volatility, and momentum
            if close[i] > r1_4h[i] and volume_filter and volatility_filter and momentum_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, volatility, and momentum
            elif close[i] < s1_4h[i] and volume_filter and volatility_filter and momentum_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below pivot or volatility drops
            pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
            if close[i] < pivot_4h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above pivot or volatility drops
            pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
            if close[i] > pivot_4h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_VolVol_Mom"
timeframe = "4h"
leverage = 1.0