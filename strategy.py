#!/usr/bin/env python3
# 12h_KAMA_Direction_RSI_ChopFilter
# Hypothesis: On 12h timeframe, use KAMA direction as primary trend signal with RSI for momentum confirmation and Chop filter to avoid ranging markets.
# Enter long when KAMA trending up, RSI > 50, and Chop < 61.8 (trending regime).
# Enter short when KAMA trending down, RSI < 50, and Chop < 61.8 (trending regime).
# Exit when KAMA direction changes or Chop > 61.8 (ranging market).
# Uses close-based KAMA and RSI to avoid look-ahead. Chop filter reduces whipsaws in ranging markets.
# Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe.

name = "12h_KAMA_Direction_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    def calculate_kama(close_prices, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volatility = np.sum(np.abs(np.diff(close_prices)), axis=0)
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close_prices)
        kama[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    # Calculate RSI (Relative Strength Index)
    def calculate_rsi(close_prices, length=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate Choppiness Index
    def calculate_choppiness(high_prices, low_prices, close_prices, length=14):
        atr = np.zeros_like(close_prices)
        tr1 = np.abs(high_prices - low_prices)
        tr2 = np.abs(np.roll(high_prices, 1) - close_prices)
        tr3 = np.abs(np.roll(low_prices, 1) - close_prices)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr = pd.Series(tr).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        
        max_high = pd.Series(high_prices).rolling(window=length, min_periods=length).max().values
        min_low = pd.Series(low_prices).rolling(window=length, min_periods=length).min().values
        
        # Avoid division by zero and invalid values
        range_hl = max_high - min_low
        atr_sum = atr * length
        chop = np.where(range_hl != 0, 100 * np.log10(atr_sum / range_hl) / np.log10(length), 50)
        return chop
    
    # Calculate KAMA, RSI, and Chop
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    rsi = calculate_rsi(close, length=14)
    chop = calculate_choppiness(high, low, close, length=14)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        kama_val = kama[i]
        kama_prev = kama[i-1]
        rsi_val = rsi[i]
        chop_val = chop[i]
        vol_ma_val = vol_ma[i]
        
        # Determine KAMA direction
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        if position == 0:
            # LONG: KAMA trending up, RSI > 50, Chop < 61.8 (trending market), Volume above average
            if kama_rising and rsi_val > 50 and chop_val < 61.8 and volume[i] > vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA trending down, RSI < 50, Chop < 61.8 (trending market), Volume above average
            elif kama_falling and rsi_val < 50 and chop_val < 61.8 and volume[i] > vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA changes direction or Chop > 61.8 (ranging market)
            if not kama_rising or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA changes direction or Chop > 61.8 (ranging market)
            if not kama_falling or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals