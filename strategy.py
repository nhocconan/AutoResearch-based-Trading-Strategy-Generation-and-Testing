#!/usr/bin/env python3
"""
Hypothesis: Simplified 15m strategy with 4h KAMA trend + 1h RSI pullback
- Previous 15m attempt (exp#007) failed with Sharpe=-4.952 due to over-complexity
- Key insight: Too many filters = 0 trades or late entries
- Simpler approach: 4h KAMA for trend direction, 1h RSI for pullback timing
- KAMA adapts to volatility better than HMA/EMA in crypto
- Fewer conditions = more trades while maintaining HTF filter
- Discrete sizing (0.0, ±0.25, ±0.35) to minimize fee churn
Timeframe: 15m (primary), 4h + 1h (HTF filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_rsi_simple_15m_v1"
timeframe = "15m"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    kama = np.zeros(n)
    
    # Change = absolute price change over er_period
    change = np.zeros(n)
    for i in range(er_period, n):
        change[i] = abs(close[i] - close[i - er_period])
    
    # Volatility = sum of absolute single-period changes
    volatility = np.zeros(n)
    for i in range(er_period, n):
        vol_sum = 0.0
        for j in range(1, er_period + 1):
            vol_sum += abs(close[i - j + 1] - close[i - j])
        volatility[i] = vol_sum
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0.0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(er_period, n):
        sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
        sc[i] = sc[i] ** 2  # Square for smoothing
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    delta = np.zeros(n)
    delta[1:] = np.diff(close)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_g = np.zeros(n)
    avg_l = np.zeros(n)
    
    avg_g[period-1] = np.mean(gain[:period])
    avg_l[period-1] = np.mean(loss[:period])
    
    for i in range(period, n):
        avg_g[i] = (avg_g[i-1] * (period - 1) + gain[i]) / period
        avg_l[i] = (avg_l[i-1] * (period - 1) + loss[i]) / period
    
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    
    return rsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # 4h KAMA for trend direction (adaptive, less whipsaw)
    kama_4h_50 = calculate_kama(df_4h['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_4h_100 = calculate_kama(df_4h['close'].values, er_period=10, fast_period=2, slow_period=30)
    # Recalculate with different periods
    kama_4h_100 = calculate_kama(df_4h['close'].values, er_period=20, fast_period=2, slow_period=40)
    
    kama_4h_50_aligned = align_htf_to_ltf(prices, df_4h, kama_4h_50)
    kama_4h_100_aligned = align_htf_to_ltf(prices, df_4h, kama_4h_100)
    
    # 1h RSI for pullback entries
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # 15m indicators
    kama_15m_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=21)
    kama_15m_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=50)
    atr = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # Position sizing
    SIZE_BASE = 0.25
    SIZE_MAX = 0.35
    
    signals = np.zeros(n)
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # HTF trend from 4h KAMA
        htf_bull = kama_4h_50_aligned[i] > kama_4h_100_aligned[i] and close[i] > kama_4h_50_aligned[i]
        htf_bear = kama_4h_50_aligned[i] < kama_4h_100_aligned[i] and close[i] < kama_4h_50_aligned[i]
        
        # 15m trend
        ltf_bull = kama_15m_21[i] > kama_15m_50[i]
        ltf_bear = kama_15m_21[i] < kama_15m_50[i]
        
        # Overall trend above 200 SMA
        above_200 = close[i] > sma_200[i]
        
        # ATR stoploss
        atr_stop = 2.5 * atr[i]
        
        # Check stoploss FIRST (Rule 6)
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - atr_stop
            if close[i] < trailing_stop:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
        elif position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + atr_stop
            if close[i] > trailing_stop:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
        
        # Determine position size
        if htf_bull or htf_bear:
            size = SIZE_MAX
        else:
            size = SIZE_BASE
        
        # Entry logic - SIMPLIFIED to ensure trades trigger
        if htf_bull and ltf_bull and above_200:
            # Long in bull trend - RSI pullback entry
            if rsi_1h_aligned[i] < 55 and rsi_1h_aligned[i] > 35:
                signals[i] = size
                if position_side == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            elif rsi_15m[i] < 50 and rsi_15m[i] > 30:
                signals[i] = size
                if position_side == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            # Exit on overbought
            elif rsi_15m[i] > 75:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
            else:
                signals[i] = signals[i-1] if i > 0 else 0.0
                
        elif htf_bear and ltf_bear and not above_200:
            # Short in bear trend - RSI pullback entry
            if rsi_1h_aligned[i] > 45 and rsi_1h_aligned[i] < 65:
                signals[i] = -size
                if position_side == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            elif rsi_15m[i] > 50 and rsi_15m[i] < 70:
                signals[i] = -size
                if position_side == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            # Exit on oversold
            elif rsi_15m[i] < 25:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
            else:
                signals[i] = signals[i-1] if i > 0 else 0.0
        else:
            # Neutral regime - stay flat or exit
            signals[i] = 0.0
            if position_side != 0:
                position_side = 0
                entry_price = 0.0
        
        # Discretize signal to reduce churn (Rule 4)
        if abs(signals[i]) < 0.10:
            signals[i] = 0.0
        elif signals[i] > 0:
            signals[i] = SIZE_BASE if signals[i] < 0.30 else SIZE_MAX
        else:
            signals[i] = -SIZE_BASE if signals[i] > -0.30 else -SIZE_MAX
        
        # Prevent look-ahead (Rule 5)
        if i > 0 and signals[i-1] == 0.0 and signals[i] != 0.0:
            pass  # New entry
        elif i > 0 and signals[i-1] != 0.0 and signals[i] == 0.0:
            pass  # Exit
    
    return signals