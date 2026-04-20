#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_Trend_RSI_Range"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === Daily KAMA Trend (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)
    er = np.zeros_like(close_1d)
    er[10:] = change[10:] / np.where(volatility[10:] > 0, volatility[10:], 1)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    # Align KAMA to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === 4h RSI (mean reversion signal) ===
    close = prices['close'].values
    delta = np.diff(close, n=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h Bollinger Bands Width (chop regime filter) ===
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_width = (4 * std20) / np.where(sma20 > 0, sma20, np.nan)  # 4*std = upper-lower
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(bb_width[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_aligned[i]
        rsi_val = rsi[i]
        bb_width_val = bb_width[i]
        close_val = close[i]
        
        # Chop regime: bb_width > 0.05 = range (mean revert), < 0.03 = trending
        if bb_width_val > 0.05:  # Range regime
            if position == 0:
                # Mean reversion: buy oversold, sell overbought
                if rsi_val < 30:
                    signals[i] = 0.25
                    position = 1
                elif rsi_val > 70:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long when RSI returns to neutral
                if rsi_val > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short when RSI returns to neutral
                if rsi_val < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:  # Trending regime
            if position == 0:
                # Trend following: buy above KAMA, sell below KAMA
                if close_val > kama_val:
                    signals[i] = 0.25
                    position = 1
                elif close_val < kama_val:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long when price crosses below KAMA
                if close_val < kama_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short when price crosses above KAMA
                if close_val > kama_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals