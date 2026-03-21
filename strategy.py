#!/usr/bin/env python3
"""
Hypothesis: 4h primary with 1d trend filter + KAMA adaptive trend + RSI timing + ATR stoploss
- 1d HMA(21) slope determines major trend direction (HTF filter)
- 4h KAMA(14) adaptive trend for entry timing (responds to volatility)
- 4h RSI(14) for pullback entries in trend direction
- 4h ATR(14) for dynamic stoploss at 2.5x
- Discrete position sizing (0.0, ±0.25, ±0.30) to minimize fee churn
- Should generate 10+ trades per symbol over 4 years with controlled DD
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_rsi_atr_4h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average for smooth trend detection"""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_kama(close, period=14, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average - adapts to market noise"""
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(period))
    volatility = abs(close_s - close_s.shift(1)).rolling(window=period, min_periods=period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = pd.Series(index=close_s.index, dtype=float)
    kama.iloc[period-1] = close_s.iloc[period-1]
    
    for i in range(period, len(close_s)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama.iloc[i-1])
    
    return kama.values

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range for stoploss"""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    return atr.values

def calculate_rsi(close, period=14):
    """Calculate RSI for entry timing"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    return rsi.values

def generate_signals(prices):
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)  # auto shift(1) for completed bars
    
    # Calculate 4h indicators (pre-compute before loop for performance)
    kama_4h = calculate_kama(close, 14)
    rsi_4h = calculate_rsi(close, 14)
    atr_4h = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_EXIT = 0.0
    
    # Track position state
    position_side = 0
    entry_price = 0.0
    stoploss_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # 1d trend filter: HMA slope direction
        if i >= 51:
            trend_1d = 1 if hma_1d_aligned[i] > hma_1d_aligned[i-1] else -1
        else:
            trend_1d = 0
        
        # 4h KAMA trend
        trend_4h = 1 if kama_4h[i] > kama_4h[i-1] else -1
        
        # RSI value
        rsi_val = rsi_4h[i]
        atr_val = atr_4h[i]
        price = close[i]
        
        # Check stoploss first (Rule 6)
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, price)
            if price < stoploss_price:
                signals[i] = SIZE_EXIT
                position_side = 0
                entry_price = 0.0
                continue
            # Trail stop: move stoploss up when price moves 1.5*ATR in profit
            trail_stop = highest_since_entry - 1.5 * atr_val
            if trail_stop > stoploss_price:
                stoploss_price = trail_stop
            signals[i] = SIZE_ENTRY
        
        elif position_side == -1:
            lowest_since_entry = min(lowest_since_entry, price)
            if price > stoploss_price:
                signals[i] = SIZE_EXIT
                position_side = 0
                entry_price = 0.0
                continue
            # Trail stop: move stoploss down when price moves 1.5*ATR in profit
            trail_stop = lowest_since_entry + 1.5 * atr_val
            if trail_stop < stoploss_price:
                stoploss_price = trail_stop
            signals[i] = -SIZE_ENTRY
        
        else:
            # No position - look for entry
            # Long: 1d uptrend + 4h uptrend + RSI pullback (not overbought)
            if trend_1d > 0 and trend_4h > 0 and rsi_val < 58 and rsi_val > 35:
                signals[i] = SIZE_ENTRY
                position_side = 1
                entry_price = price
                stoploss_price = entry_price - 2.5 * atr_val
                highest_since_entry = price
            
            # Short: 1d downtrend + 4h downtrend + RSI pullback (not oversold)
            elif trend_1d < 0 and trend_4h < 0 and rsi_val > 42 and rsi_val < 65:
                signals[i] = -SIZE_ENTRY
                position_side = -1
                entry_price = price
                stoploss_price = entry_price + 2.5 * atr_val
                lowest_since_entry = price
    
    return signals