#!/usr/bin/env python3
"""
Experiment #824: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend with RSI Entries

Hypothesis: KAMA (Kaufman Adaptive Moving Average) outperforms HMA/EMA on BTC/ETH
because it adapts to market regime automatically. In trending markets, KAMA follows
price closely. In choppy/ranging markets (common in 2022-2024), KAMA flattens and
avoids whipsaw entries. Combined with 1w ultra-HTF bias + 1d trend + 12h entries.

Key innovations:
1. 1w HMA(21) for ultra-HTF regime bias (bull/bear market identification)
2. 1d KAMA(10,2,30) for adaptive HTF trend direction
3. 12h KAMA crossover (fast/slow) for entry timing
4. 12h RSI(14) with loose thresholds (40/60) for entry confirmation
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Entry conditions (LOOSE to ensure ≥30 trades/train, ≥3/test):
- LONG: 1w HMA bull + 1d KAMA bull + (12h KAMA cross OR RSI<45)
- SHORT: 1w HMA bear + 1d KAMA bear + (12h KAMA cross OR RSI>55)

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_adaptive_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio.
    ER = |net change| / sum of absolute changes over period
    High ER (trending) → fast smoothing constant
    Low ER (choppy) → slow smoothing constant
    """
    n = len(close)
    if n < slow_period + er_period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(er_period, n):
        net_change = abs(close[i] - close[i - er_period])
        sum_changes = 0.0
        for j in range(1, er_period + 1):
            sum_changes += abs(close[i - j + 1] - close[i - j])
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Initialize KAMA at SMA of first slow_period bars
    kama[slow_period] = np.mean(close[:slow_period + 1])
    
    for i in range(slow_period + 1, n):
        if not np.isnan(er[i]):
            # Adaptive smoothing constant
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 12h indicators
    kama_12h_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=10)
    kama_12h_slow = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_12h_fast[i]) or np.isnan(kama_12h_slow[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === ULTRA-HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === HTF TREND (1d KAMA) ===
        htf_1d_bull = close[i] > kama_1d_aligned[i]
        htf_1d_bear = close[i] < kama_1d_aligned[i]
        
        # === 12h KAMA CROSSOVER ===
        kama_crossover_long = False
        kama_crossover_short = False
        if i > 0 and not np.isnan(kama_12h_fast[i-1]) and not np.isnan(kama_12h_slow[i-1]):
            kama_crossover_long = (kama_12h_fast[i-1] <= kama_12h_slow[i-1]) and (kama_12h_fast[i] > kama_12h_slow[i])
            kama_crossover_short = (kama_12h_fast[i-1] >= kama_12h_slow[i-1]) and (kama_12h_fast[i] < kama_12h_slow[i])
        
        # === 12h KAMA TREND ===
        kama_12h_bull = kama_12h_fast[i] > kama_12h_slow[i]
        kama_12h_bear = kama_12h_fast[i] < kama_12h_slow[i]
        
        # === RSI CONDITIONS (LOOSE for more trades) ===
        rsi_oversold = rsi_14[i] < 45.0
        rsi_overbought = rsi_14[i] > 55.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: 1w bull + 1d bull + (12h KAMA cross OR RSI oversold OR 12h KAMA bull)
        if htf_1w_bull and htf_1d_bull:
            if rsi_oversold or kama_crossover_long or kama_12h_bull:
                if rsi_extreme_oversold or kama_crossover_long:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 1w bear + 1d bear + (12h KAMA cross OR RSI overbought OR 12h KAMA bear)
        elif htf_1w_bear and htf_1d_bear:
            if rsi_overbought or kama_crossover_short or kama_12h_bear:
                if rsi_extreme_overbought or kama_crossover_short:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals