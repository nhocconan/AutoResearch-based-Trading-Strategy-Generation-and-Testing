#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA + RSI(14) + Choppiness Index regime filter
# KAMA adapts to market efficiency - slow in ranging, fast in trending markets
# RSI(14) provides mean reversion signals in ranging markets and momentum in trends
# Choppiness Index filters regime: >61.8 = range (mean revert), <38.2 = trend (follow momentum)
# Designed for 4h timeframe with target 20-50 trades/year to minimize fee drag
# Works in both bull and bear markets via regime adaptation

name = "4h_kama_rsi_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Adaptive Moving Average) parameters
    er_length = 10
    fast_sc = 2
    slow_sc = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    sum_change = pd.Series(change).rolling(window=er_length, min_periods=1).sum().values
    sum_abs_change = pd.Series(abs_change).rolling(window=er_length, min_periods=1).sum().values
    er = np.where(sum_abs_change != 0, sum_change / sum_abs_change, 0)
    
    # Calculate Smoothing Constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = np.where(sum_atr != 0, 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14), 50)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Regime filter from Choppiness Index
        ranging = chop[i] > 61.8
        trending = chop[i] < 38.2
        
        # KAMA trend
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit when RSI overbought or trend changes against position
            if rsi[i] > 70 or (ranging and not price_above_kama):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit when RSI oversold or trend changes against position
            if rsi[i] < 30 or (ranging and not price_below_kama):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Entry conditions based on regime
            if ranging and vol_confirm:
                # In ranging market: mean reversion at RSI extremes
                if rsi[i] < 30 and price_above_kama and i > 0 and rsi[i-1] >= 30:
                    position = 1
                    signals[i] = 0.25
                elif rsi[i] > 70 and price_below_kama and i > 0 and rsi[i-1] <= 70:
                    position = -1
                    signals[i] = -0.25
            elif trending and vol_confirm:
                # In trending market: momentum with KAMA
                if rsi[i] > 50 and price_above_kama and i > 0 and rsi[i-1] <= 50:
                    position = 1
                    signals[i] = 0.25
                elif rsi[i] < 50 and price_below_kama and i > 0 and rsi[i-1] >= 50:
                    position = -1
                    signals[i] = -0.25
    
    return signals