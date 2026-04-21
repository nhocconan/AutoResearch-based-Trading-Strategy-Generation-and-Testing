#!/usr/bin/env python3
"""
4h_KAMA_Regime_Trend_ATRStop
Hypothesis: 4h KAMA trend direction filtered by 1d EMA200 trend and ATR-based volatility regime.
Enter long when 4h KAMA turns up with 1d EMA200 uptrend and low volatility regime (ATR ratio < 1.2).
Enter short when 4h KAMA turns down with 1d EMA200 downtrend and low volatility regime.
Exit on ATR(14) trailing stop (2.0*ATR) or opposite KAMA signal.
Designed for low trade frequency (<25 trades/year) to minimize fee drag.
Works in bull/bear via 1d trend alignment and volatility regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for KAMA, 1d for trend filter)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 200:
        return np.zeros(n)
    
    # === 4h KAMA for trend direction ===
    close_4h = df_4h['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_4h, 10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_4h)), axis=0)  # 10-period sum of absolute changes
    # Pad volatility array to match length
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close_4h, np.nan)
    kama[9] = close_4h[9]  # Start after 10 periods
    for i in range(10, len(close_4h)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    
    # === 1d EMA200 for HTF trend filter ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === ATR (14-period) for stoploss and volatility regime ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime filter)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma  # < 1.2 = low volatility regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) 
            or np.isnan(atr[i]) or np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # KAMA turning point: cross above/below KAMA
            kama_cross_up = i > 10 and close[i-1] <= kama_aligned[i-1] and price > kama_aligned[i]
            kama_cross_down = i > 10 and close[i-1] >= kama_aligned[i-1] and price < kama_aligned[i]
            
            # Trend filter: price relative to 1d EMA200
            uptrend = price > ema_200_1d_aligned[i]
            downtrend = price < ema_200_1d_aligned[i]
            
            # Volatility regime filter: low volatility (ATR ratio < 1.2)
            low_vol = atr_ratio[i] < 1.2
            
            # Entry logic
            if kama_cross_up and uptrend and low_vol:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif kama_cross_down and downtrend and low_vol:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below KAMA (trend change)
            elif price < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above KAMA (trend change)
            elif price > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Regime_Trend_ATRStop"
timeframe = "4h"
leverage = 1.0