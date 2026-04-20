#!/usr/bin/env python3
"""
12h KAMA Direction + RSI + Chop Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets.
Combined with RSI for momentum confirmation and Choppiness Index to filter ranging conditions, this strategy avoids whipsaws.
Trades only when KAMA slope confirms trend, RSI is not extreme, and market is trending (CHOP < 61.8).
Targets 12-37 trades per year by requiring confluence of three filters, reducing false signals and fee impact.
Works in bull markets via trend continuation and in bear markets via avoidance of false breakouts during ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for KAMA, RSI, and Choppiness Index
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(close_daily - np.roll(close_daily, 10))
    change[0:10] = np.abs(close_daily[0:10] - close_daily[0])
    volatility = np.sum(np.abs(np.diff(close_daily, prepend=close_daily[0])), axis=0) if False else None
    # Correct volatility calculation: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close_daily)
    for i in range(len(close_daily)):
        if i < 10:
            volatility[i] = np.sum(np.abs(np.diff(close_daily[0:i+1]))) if i > 0 else 0
        else:
            volatility[i] = np.sum(np.abs(np.diff(close_daily[i-9:i+1])))
    er = np.zeros_like(close_daily)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_daily)
    kama[0] = close_daily[0]
    for i in range(1, len(close_daily)):
        kama[i] = kama[i-1] + sc[i] * (close_daily[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (CHOP) over 14 periods
    atr = np.zeros_like(close_daily)
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_daily).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_daily).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) != 0, 100 * np.log10(atr / (max_high - min_low)) / np.log10(14), 50)
    
    # Align daily indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_daily, chop)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma
        
        # KAMA slope: positive if current > prior, negative if current < prior
        kama_slope = kama_val - kama_aligned[i-1] if i > 0 else 0
        
        if position == 0:
            # Long: price above KAMA, RSI not overbought, market trending (CHOP < 61.8), volume confirmation
            if price > kama_val and rsi_val < 70 and chop_val < 61.8 and vol_ok and kama_slope > 0:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI not oversold, market trending (CHOP < 61.8), volume confirmation
            elif price < kama_val and rsi_val > 30 and chop_val < 61.8 and vol_ok and kama_slope < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI overbought
            if price < kama_val or rsi_val > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI oversold
            if price > kama_val or rsi_val < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Direction_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0