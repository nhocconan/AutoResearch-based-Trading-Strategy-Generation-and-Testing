#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_1dVolatility_Regime_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volatility regime and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility regime
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), np.abs(high_1d[0:-1] - close_1d[1:]))
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR ratio: current ATR / 50-period average ATR (volatility regime filter)
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / atr_ma_50
    # Low volatility regime: ATR ratio < 0.8 (calm market)
    # High volatility regime: ATR ratio > 1.2 (volatile market)
    low_vol = atr_ratio < 0.8
    high_vol = atr_ratio > 1.2
    
    # Calculate KAMA on 4h close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(close)
    er[10:] = change[10:] / volatility[10:]
    er[volatility == 0] = 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align 1d indicators to 4h
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol)
    high_vol_aligned = align_htf_to_ltf(prices, df_1d, high_vol)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(low_vol_aligned[i]) or 
            np.isnan(high_vol_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility regime: mean reversion at KAMA
        # High volatility regime: trend following
        
        if position == 0:
            # Enter long in low vol when price < KAMA (oversold)
            # Enter short in low vol when price > KAMA (overbought)
            # In high vol, follow price > KAMA for long, price < KAMA for short
            if low_vol_aligned[i]:
                # Mean reversion: fade extreme
                if close[i] < kama[i] * 0.998:  # 0.2% below KAMA
                    signals[i] = 0.25
                    position = 1
                elif close[i] > kama[i] * 1.002:  # 0.2% above KAMA
                    signals[i] = -0.25
                    position = -1
            else:  # high vol or neutral
                # Trend following
                if close[i] > kama[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price crosses back above KAMA in low vol, or below in high vol
            # Or use ATR-based stop
            if low_vol_aligned[i]:
                if close[i] > kama[i] * 1.001:  # 0.1% above KAMA
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # high vol
                if close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back below KAMA in low vol, or above in high vol
            if low_vol_aligned[i]:
                if close[i] < kama[i] * 0.999:  # 0.1% below KAMA
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # high vol
                if close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals