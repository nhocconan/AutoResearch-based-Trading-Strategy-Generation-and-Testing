#!/usr/bin/env python3
"""
Experiment #092: 12h Primary + 1d HTF — KAMA Adaptive Trend + RSI Filter

Hypothesis: KAMA (Kaufman Adaptive Moving Average) outperforms HMA/EMA in mixed
regime markets (2021-2024 includes bull, bear, and range). KAMA adapts its
smoothing based on market efficiency ratio (ER), reducing whipsaw in chop while
tracking trends tightly. 

Key innovations vs #086:
1. KAMA instead of HMA — adapts smoothing to market regime automatically
2. Efficiency Ratio (ER) as additional filter — only trade when ER > 0.3 (some trend)
3. 1d KAMA for bias (not HMA) — more adaptive to regime changes
4. RSI thresholds 40/60 (slightly tighter than 35/65 for better quality)
5. Volume confirmation: taker_buy_volume ratio > 0.45 for longs, < 0.55 for shorts

Target: Sharpe>0.351 (beat current best), DD>-40%, trades>=30 train, >=3 test
Position size: 0.30 (30% of capital, discrete levels)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_er_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market Efficiency Ratio (ER)
    ER = |net change| / sum of absolute changes over period
    High ER = trending (fast smoothing), Low ER = choppy (slow smoothing)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        net_change = abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA at first valid point
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_er(close, period=10):
    """Efficiency Ratio - measures trend efficiency (0=chop, 1=strong trend)"""
    n = len(close)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        net_change = abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    return er

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for major trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=20, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (12h) indicators
    kama_fast = calculate_kama(close, period=10, fast_period=2, slow_period=20)
    kama_slow = calculate_kama(close, period=30, fast_period=2, slow_period=40)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    er = calculate_er(close, period=14)
    
    # Volume ratio (taker buy / total volume)
    vol_ratio = np.zeros(n)
    vol_ratio[:] = np.nan
    for i in range(n):
        if volume[i] > 1e-10:
            vol_ratio[i] = taker_buy_vol[i] / volume[i]
        else:
            vol_ratio[i] = 0.5
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for 12h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(kama_1d_aligned[i]) or np.isnan(er[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d KAMA) ===
        htf_bull = close[i] > kama_1d_aligned[i]
        htf_bear = close[i] < kama_1d_aligned[i]
        
        # === 12h TREND (KAMA crossover) ===
        kama_cross_bull = kama_fast[i] > kama_slow[i]
        kama_cross_bear = kama_fast[i] < kama_slow[i]
        
        # === EFFICIENCY RATIO FILTER ===
        # Only trade when there's some trend (ER > 0.3)
        er_ok = er[i] > 0.30
        
        # === RSI FILTER (moderate - ensure trades but filter extremes) ===
        rsi_ok_long = rsi[i] > 40.0
        rsi_ok_short = rsi[i] < 60.0
        
        # === VOLUME CONFIRMATION ===
        # Longs: more buyer aggression (vol_ratio > 0.45)
        # Shorts: more seller aggression (vol_ratio < 0.55)
        vol_ok_long = vol_ratio[i] > 0.45
        vol_ok_short = vol_ratio[i] < 0.55
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + 12h KAMA cross bull + ER>0.3 + RSI>40 + vol confirm
        # SHORT: 1d bear + 12h KAMA cross bear + ER>0.3 + RSI<60 + vol confirm
        desired_signal = 0.0
        
        if htf_bull and kama_cross_bull and er_ok and rsi_ok_long and vol_ok_long:
            desired_signal = SIZE
        elif htf_bear and kama_cross_bear and er_ok and rsi_ok_short and vol_ok_short:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals