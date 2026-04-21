#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop regime + Volume spike
# Long when KAMA rising, RSI > 50, Chop > 61.8 (trending), Volume > 1.5x 20-day avg
# Short when KAMA falling, RSI < 50, Chop > 61.8, Volume > 1.5x 20-day avg
# Exit when KAMA direction reverses or Chop < 38.2 (range)
# Uses daily timeframe for signals, reduces trade frequency to avoid fee drag
# KAMA adapts to market efficiency, RSI filters momentum, Chop confirms trend strength
# Volume spike confirms institutional participation
# Target: 15-25 trades/year by requiring multiple confluence factors

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppy Index(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    atr = np.zeros_like(close_1d)
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (max_high - min_low)) / np.log10(14)
    
    # Calculate 20-day volume average
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to daily timeframe (no alignment needed for 1d->1d)
    kama_aligned = kama
    rsi_aligned = rsi
    chop_aligned = chop
    vol_ma_aligned = vol_ma_1d
    
    # Calculate daily price change for KAMA direction
    kama_dir = np.diff(kama_aligned, prepend=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        kama_dir_val = kama_dir[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        vol_ma = vol_ma_aligned[i]
        
        # Volume confirmation: current daily volume > 1.5x 20-day average
        volume_confirm = df_1d['volume'].iloc[i] > 1.5 * vol_ma
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, Chop > 61.8 (trending), volume confirmation
            if kama_dir_val > 0 and rsi_val > 50 and chop_val > 61.8 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI < 50, Chop > 61.8 (trending), volume confirmation
            elif kama_dir_val < 0 and rsi_val < 50 and chop_val > 61.8 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if KAMA falls or Chop < 38.2 (range)
                if kama_dir_val <= 0 or chop_val < 38.2:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if KAMA rises or Chop < 38.2 (range)
                if kama_dir_val >= 0 or chop_val < 38.2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_Volume"
timeframe = "1d"
leverage = 1.0