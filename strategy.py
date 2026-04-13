#!/usr/bin/env python3
"""
4h_1d_kama_rsi_volatility
Hypothesis: Use 1-day KAMA trend direction combined with RSI mean-reversion and volatility expansion to capture trend-following entries with pullback entries in both bull and bear markets.
Volatility filter (ATR expansion) avoids choppy markets. Targets 20-30 trades/year to minimize fee drag.
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
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend indicator
    def calculate_kama(close, er_len=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close, prepend=close[0]))
        for i in range(1, len(close)):
            volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        
        er = np.zeros_like(close)
        for i in range(er_len, len(close)):
            if volatility[i-er_len] != 0:
                er[i] = change[i] / volatility[i-er_len]
            else:
                er[i] = 0
        
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # RSI (Relative Strength Index)
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.zeros_like(close)
        rsi = np.zeros_like(close)
        for i in range(period, len(close)):
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100
        return rsi
    
    # ATR (Average True Range) for volatility
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[0] = tr[0]
        for i in range(1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    kama_1d = calculate_kama(close_1d, 10, 2, 30)
    rsi_1d = calculate_rsi(close_1d, 14)
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Volatility expansion: current ATR > 1.5x 20-period average
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    vol_expansion = atr_1d > (atr_ma_20 * 1.5)
    
    # Align all signals to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_expansion_aligned = align_htf_to_ltf(prices, df_1d, vol_expansion.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_expansion_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below KAMA
        above_kama = close[i] > kama_aligned[i]
        below_kama = close[i] < kama_aligned[i]
        
        # Entry conditions: RSI mean-reversion with volatility expansion
        long_entry = below_kama and (rsi_aligned[i] < 30) and vol_expansion_aligned[i] > 0.5
        short_entry = above_kama and (rsi_aligned[i] > 70) and vol_expansion_aligned[i] > 0.5
        
        # Exit conditions: RSI returns to neutral zone
        exit_long = position == 1 and rsi_aligned[i] > 50
        exit_short = position == -1 and rsi_aligned[i] < 50
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_kama_rsi_volatility"
timeframe = "4h"
leverage = 1.0