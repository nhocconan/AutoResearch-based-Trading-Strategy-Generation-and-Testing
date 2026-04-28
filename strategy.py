#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Pullback_Volume_Regime
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) direction filter (fast/slow crossover) 
combined with RSI pullback to KAMA and volume regime filter to capture mean-reversion 
within trending markets. Designed for low trade frequency (20-40/year) to minimize fee drag 
while working in both bull (buy dips in uptrend) and bear (sell rallies in downtrend). 
Targets 80-160 total trades over 4 years.
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
    
    # Get 1d data for regime filter (choppiness)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR14
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Absolute price change over 14 periods
    abs_change = np.abs(close_1d - np.roll(close_1d, 14))
    abs_change = np.concatenate([np.full(14, np.nan), abs_change[14:]])
    
    # Sum of absolute changes over 14 periods
    sum_abs_change = pd.Series(abs_change).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(sum(tr14) / sum(abs_change14)) / log10(14)
    chop_raw = 100 * np.log10(atr_14 * 14 / sum_abs_change) / np.log10(14)
    chop_1d = pd.Series(chop_raw).fillna(50).values  # neutral when undefined
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 4h data for KAMA and RSI
    # Calculate KAMA (ER=10, SC=2,30)
    # Efficiency Ratio
    change = np.abs(np.diff(close, 10))
    change = np.concatenate([np.full(10, np.nan), change])
    
    # Volatility (sum of absolute changes)
    vol = np.sum(np.abs(np.diff(close, 1)), axis=0) if len(close) > 1 else np.array([np.nan])
    vol = pd.Series(np.abs(np.diff(close, 1))).rolling(window=10, min_periods=1).sum().values
    vol = np.concatenate([np.full(9, np.nan), vol[9:]])
    
    # Avoid division by zero
    er = np.divide(change, vol, out=np.zeros_like(change), where=vol!=0)
    
    # Smoothing Constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # Volume confirmation: >1.5x 48-period MA
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 48)  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI pullback conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_48[i])
        
        # Regime filter: only trade in ranging markets (chop > 50)
        ranging_market = chop_1d_aligned[i] > 50
        
        # Entry conditions
        long_entry = price_above_kama and rsi_oversold and vol_confirm and ranging_market
        short_entry = price_below_kama and rsi_overbought and vol_confirm and ranging_market
        
        # Exit conditions: opposite RSI extreme or KAMA crossover
        long_exit = rsi[i] > 70 or close[i] < kama[i]
        short_exit = rsi[i] < 30 or close[i] > kama[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Direction_RSI_Pullback_Volume_Regime"
timeframe = "4h"
leverage = 1.0