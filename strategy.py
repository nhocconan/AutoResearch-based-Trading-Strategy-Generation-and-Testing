#!/usr/bin/env python3
"""
Experiment #1631: 6h Primary + 1w/1d HTF — Multi-Day Momentum with Weekly Trend Filter

Hypothesis: 6h timeframe captures multi-day momentum swings better than 4h (less noise) 
and 12h (more responsive). Weekly HMA provides strong trend bias while 6h KAMA + ROC 
captures momentum shifts. Simpler entry logic (no complex regime filters) should generate 
more trades than failed 6h experiments (#1620, #1623, #1627).

Key design choices based on failure analysis:
1. NO regime filters (CHOP, etc.) - these reduced trades to near-zero on 6h
2. RSI(7) not RSI(14) - faster crosses = more entry signals
3. ROC(10) for momentum - simpler than Fisher, proven on higher TFs
4. KAMA for adaptive trend - less lag than EMA, adjusts to volatility
5. Weekly HMA(21) for trend bias - strong filter, only trade with weekly trend
6. Discrete signal sizes: 0.25 base, 0.30 strong (with weekly confirmation)
7. 2.5x ATR trailing stoploss via signal→0

Entry logic (SIMPLE to guarantee ≥30 trades/train):
- LONG: 1w HMA bullish + 6h KAMA rising + ROC(10) > 0 + RSI(7) > 50
- SHORT: 1w HMA bearish + 6h KAMA falling + ROC(10) < 0 + RSI(7) < 50
- Strong signal: + 1d HMA confirmation in same direction

Why this beats mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- Simpler entry = more trades = better statistical edge
- RSI(7) crosses 50 much more often than RSI(14) crosses 30/70
- Weekly trend filter prevents counter-trend trades in strong moves
- KAMA adapts to volatility changes better than fixed EMA

Target: Sharpe>0.6, trades≥30 train, trades≥5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_momentum_weekly_kama_roc_rsi7_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio
    Less lag in trends, more smoothing in chop
    """
    n = len(close)
    if n < slow_period + er_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        if not np.isnan(close[i]):
            price_change = abs(close[i] - close[i - er_period])
            volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            if volatility > 1e-10:
                er[i] = price_change / volatility
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] > 1e-10:
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100.0
    
    return roc

def calculate_rsi(close, period=7):
    """Relative Strength Index - faster period for more signals"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
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
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    kama_6h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    roc_10 = calculate_roc(close, period=10)
    rsi_7 = calculate_rsi(close, period=7)
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
    
    # Warmup period
    min_bars = 60
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_6h[i]) or np.isnan(roc_10[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1w HMA bias - PRIMARY) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === TREND CONFIRMATION (1d HMA - SECONDARY) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === MOMENTUM (ROC) ===
        roc_positive = roc_10[i] > 0
        roc_negative = roc_10[i] < 0
        
        # === KAMA TREND (adaptive) ===
        kama_rising = False
        kama_falling = False
        if i >= 3 and not np.isnan(kama_6h[i-3]):
            kama_rising = kama_6h[i] > kama_6h[i-3]
            kama_falling = kama_6h[i] < kama_6h[i-3]
        
        # === RSI TIMING (fast 7-period) ===
        rsi_above_50 = rsi_7[i] > 50
        rsi_below_50 = rsi_7[i] < 50
        
        # === ENTRY LOGIC (SIMPLE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: Weekly bullish + KAMA rising + ROC positive + RSI > 50
        if price_above_1w and kama_rising and roc_positive and rsi_above_50:
            # Strong if daily also confirms
            if price_above_1d:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: Weekly bearish + KAMA falling + ROC negative + RSI < 50
        elif price_below_1w and kama_falling and roc_negative and rsi_below_50:
            # Strong if daily also confirms
            if price_below_1d:
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