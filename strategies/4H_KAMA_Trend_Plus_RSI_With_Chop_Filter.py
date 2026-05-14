#!/usr/bin/env python3
"""
4H_KAMA_Trend_Plus_RSI_With_Chop_Filter
Hypothesis: KAMA adapts to market noise, providing robust trend direction. 
RSI(14) provides overbought/oversold signals within the trend. 
Choppiness index (14) filters ranging markets (CHOP > 61.8) to avoid false signals.
Trades only when KAMA trend aligns with RSI extremes and market is trending (CHOP < 61.8).
Designed for low turnover (~25-35 trades/year) to minimize fee drag in 2025 ranging markets.
"""

name = "4H_KAMA_Trend_Plus_RSI_With_Chop_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # === KAMA Calculation (ER=10, Fast=2, Slow=30) ===
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14) ===
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = np.where(sum_atr != 0, 
                    100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14), 
                    50)
    
    # === Load Daily Trend Filter (EMA 50) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50  # covers EMA50, RSI, CHOP, KAMA
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        trending_market = chop[i] < 61.8  # Chop < 61.8 = trending
        
        if position == 0:
            # Long: Price above KAMA (uptrend) + RSI oversold + trending market + volume spike
            if price_above_kama and rsi_oversold and trending_market and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price below KAMA (downtrend) + RSI overbought + trending market + volume spike
            elif price_below_kama and rsi_overbought and trending_market and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price crosses below KAMA OR RSI overbought (take profit in uptrend)
                if close[i] < kama[i] or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above KAMA OR RSI oversold (take profit in downtrend)
                if close[i] > kama[i] or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals