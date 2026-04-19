#!/usr/bin/env python3
# 12h_1D_KAMA_RSI_Chop_Filter
# Hypothesis: 12h KAMA trend direction combined with RSI momentum and 1d chop regime filter
# KAMA adapts to market noise, reducing whipsaws in choppy markets
# RSI(14) > 50 for long, < 50 for short ensures momentum alignment
# 1d Choppiness Index > 61.8 filters for ranging markets (mean reversion opportunity)
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear via adaptive trend filter and regime-aware momentum

name = "12h_1D_KAMA_RSI_Chop_Filter"
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
    
    # KAMA (Kaufman Adaptive Moving Average) - trend identification
    def calculate_kama(close, er_length=10, fast_ema=2, slow_ema=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        abs_change = np.abs(np.diff(close, prepend=close[0]))
        # Alternative calculation for efficiency ratio
        diff = np.abs(close - np.roll(close, er_length))
        diff[0] = 0
        vol = np.zeros_like(close)
        for i in range(1, len(close)):
            vol[i] = vol[i-1] + np.abs(close[i] - close[i-1])
        
        er = np.zeros_like(close)
        er[:er_length] = 0
        for i in range(er_length, len(close)):
            if vol[i] != 0:
                er[i] = diff[i] / vol[i]
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        
        return kama
    
    # RSI calculation
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Choppiness Index calculation
    def calculate_chop(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # ATR
        atr = np.zeros_like(close)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Highest high and lowest low over period
        hh = np.zeros_like(high)
        ll = np.zeros_like(low)
        for i in range(period-1, len(high)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if hh[i] != ll[i]:
                log_sum = np.log10(np.sum(tr[i-period+1:i+1]) / (hh[i] - ll[i]))
                chop[i] = 100 * log_sum / np.log10(period)
            else:
                chop[i] = 50
        
        return chop
    
    # 12h data for KAMA and RSI
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    kama_12h = calculate_kama(df_12h['close'].values)
    rsi_12h = calculate_rsi(df_12h['close'].values)
    
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # 1d data for Choppiness Index (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in choppy markets (Chop > 61.8) for mean reversion
        choppy_market = chop_1d_aligned[i] > 61.8
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, in choppy market, volume confirmation
            if (close[i] > kama_12h_aligned[i] and 
                rsi_12h_aligned[i] > 50 and 
                choppy_market and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, in choppy market, volume confirmation
            elif (close[i] < kama_12h_aligned[i] and 
                  rsi_12h_aligned[i] < 50 and 
                  choppy_market and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or RSI < 45
            if (close[i] < kama_12h_aligned[i]) or (rsi_12h_aligned[i] < 45):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or RSI > 55
            if (close[i] > kama_12h_aligned[i]) or (rsi_12h_aligned[i] > 55):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals