#!/usr/bin/env python3
"""
Experiment #011: 12h KAMA trend + 1d HMA filter + RSI timing + ATR stoploss
Hypothesis: 12h primary captures medium-term trends, 1d HTF filters major direction,
KAMA adapts to volatility better than EMA, RSI prevents chasing tops/bottoms.
Position sizing: 0.25 base, reduce to 0.125 at 2R profit, stop at 2.5*ATR
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_hma_rsi_12h_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if i < period:
            kama[i] = close[i]
            continue
        
        # Efficiency Ratio
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        er = signal / noise if noise > 0 else 0.0
        
        # Smoothing constant
        sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
        
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_hma(close, period=21):
    """Hull Moving Average for smooth trend"""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    hma = 2 * wma_half - wma_full
    hma = hma.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)  # auto shift(1)
    
    # Calculate 12h indicators
    kama_fast = calculate_kama(close, period=10, fast=2, slow=30)
    kama_slow = calculate_kama(close, period=20, fast=2, slow=30)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss/takeprofit
    entry_price = np.zeros(n)
    position_side = np.zeros(n)
    peak_price = np.zeros(n)
    
    for i in range(50, n):
        # HTF trend filter (1d HMA)
        htf_trend = 1.0 if hma_1d_aligned[i] > hma_1d_aligned[i - 5] else -1.0
        
        # 12h KAMA crossover
        kama_cross = 1.0 if kama_fast[i] > kama_slow[i] else -1.0
        
        # RSI filter - don't chase extremes
        rsi_ok_long = rsi[i] < 70 and rsi[i] > 30
        rsi_ok_short = rsi[i] < 70 and rsi[i] > 30
        
        # Current position tracking
        if i > 0:
            entry_price[i] = entry_price[i - 1]
            position_side[i] = position_side[i - 1]
            peak_price[i] = peak_price[i - 1]
        
        # Stoploss check (2.5 * ATR)
        if position_side[i] == 1.0 and entry_price[i] > 0:
            if close[i] < entry_price[i] - 2.5 * atr[i]:
                signals[i] = 0.0
                position_side[i] = 0.0
                entry_price[i] = 0.0
                continue
        
        if position_side[i] == -1.0 and entry_price[i] > 0:
            if close[i] > entry_price[i] + 2.5 * atr[i]:
                signals[i] = 0.0
                position_side[i] = 0.0
                entry_price[i] = 0.0
                continue
        
        # Take profit check (2R = 2 * 2.5 * ATR = 5 * ATR)
        if position_side[i] == 1.0 and entry_price[i] > 0:
            if close[i] > entry_price[i] + 5.0 * atr[i]:
                signals[i] = SIZE_HALF
                continue
        
        if position_side[i] == -1.0 and entry_price[i] > 0:
            if close[i] < entry_price[i] - 5.0 * atr[i]:
                signals[i] = -SIZE_HALF
                continue
        
        # Entry logic - align HTF trend with LTF signal
        if htf_trend > 0 and kama_cross > 0 and rsi_ok_long:
            signals[i] = SIZE_BASE
            if position_side[i] != 1.0:
                entry_price[i] = close[i]
                position_side[i] = 1.0
                peak_price[i] = close[i]
        elif htf_trend < 0 and kama_cross < 0 and rsi_ok_short:
            signals[i] = -SIZE_BASE
            if position_side[i] != -1.0:
                entry_price[i] = close[i]
                position_side[i] = -1.0
                peak_price[i] = close[i]
        else:
            # Hold current position or flat
            if position_side[i] == 0.0:
                signals[i] = 0.0
    
    return signals