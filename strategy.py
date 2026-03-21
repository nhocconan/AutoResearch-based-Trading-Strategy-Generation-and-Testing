#!/usr/bin/env python3
"""
EXPERIMENT #013 - 1h MACD Entry with 6h KAMA Trend Filter
==========================================================
Hypothesis: Using 6h KAMA (Kaufman Adaptive Moving Average) for trend direction
combined with 1h MACD histogram entries will capture adaptive trend changes better
than fixed-period EMAs. KAMA adjusts speed based on market noise/volatility.

Key innovations:
- 6h KAMA trend (adaptive to market efficiency - faster in trends, slower in chop)
- 1h MACD histogram for momentum entry signals (not just crossover)
- 1h RSI(14) filter to avoid overbought/oversold entries
- ATR(14) trailing stoploss - signal→0 when stopped out
- Discrete position sizing (0.0, ±0.25, ±0.30) to minimize fee churn

Different from failed strategies:
- Not 4h HMA (failed #012)
- Not Daily filter (failed #001, #004, #006, #007, #008, #010)
- Not Supertrend (failed #003, #007, #011)
- Not Donchian (failed #009)
- Not KAMA with Daily (failed #008, #010) - THIS USES 6h KAMA (NEW!)
- NEW: 6h KAMA + 1h MACD histogram + RSI filter combination
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_6h_kama_macd_rsi_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close: np.ndarray, er_period: int = 10, fast_period: int = 2, slow_period: int = 30) -> np.ndarray:
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - moves fast in trends, slow in chop
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        er[i] = signal / noise if noise > 0 else 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    ema_fast = close_s.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_s.ewm(span=slow, adjust=False, min_periods=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line.values, signal_line.values, histogram.values


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    
    avg_gain = gain_s.ewm(span=period, adjust=False, min_periods=period).mean()
    avg_loss = loss_s.ewm(span=period, adjust=False, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === LOAD 6h HTF DATA ONCE BEFORE LOOP (CRITICAL RULE #1) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    
    # Calculate 6h KAMA for adaptive trend direction
    kama_6h = calculate_kama(close_6h, er_period=10, fast_period=2, slow_period=30)
    
    # Align 6h KAMA to 1h timeframe (auto shift(1) for completed bars only)
    kama_6h_aligned = align_htf_to_ltf(prices, df_6h, kama_6h)
    
    # === CALCULATE 1h INDICATORS (vectorized before loop) ===
    rsi_1h = calculate_rsi(close, 14)
    atr_1h = calculate_atr(high, low, close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    
    # KAMA on 1h for additional trend confirmation
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # === GENERATE SIGNALS WITH STOPLOSS LOGIC ===
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30  # 30% position size on entry
    SIZE_HALF = 0.15   # 15% position size for partial exit
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    partial_exit_done = False
    
    min_bars = 100  # Ensure enough data for all indicators
    
    for i in range(min_bars, n):
        # Skip if any indicator is NaN
        if (np.isnan(kama_6h_aligned[i]) or np.isnan(rsi_1h[i]) or 
            np.isnan(atr_1h[i]) or np.isnan(macd_hist[i]) or np.isnan(kama_1h[i])):
            signals[i] = 0.0
            position_side = 0
            continue
        
        # 6h trend direction (price vs KAMA)
        kama_6h_val = kama_6h_aligned[i]
        is_uptrend_6h = close[i] > kama_6h_val
        is_downtrend_6h = close[i] < kama_6h_val
        
        # 1h trend confirmation
        kama_1h_val = kama_1h[i]
        is_uptrend_1h = close[i] > kama_1h_val if not np.isnan(kama_1h_val) else False
        is_downtrend_1h = close[i] < kama_1h_val if not np.isnan(kama_1h_val) else False
        
        # MACD histogram momentum
        macd_histogram = macd_hist[i]
        macd_histogram_prev = macd_hist[i-1] if i > 0 else 0.0
        
        # MACD bullish/bearish momentum
        macd_bullish = macd_histogram > 0 and macd_histogram > macd_histogram_prev
        macd_bearish = macd_histogram < 0 and macd_histogram < macd_histogram_prev
        
        # RSI filter
        rsi = rsi_1h[i]
        rsi_oversold = rsi < 40
        rsi_overbought = rsi > 60
        rsi_neutral = 40 <= rsi <= 60
        
        # ATR for stoploss
        atr = atr_1h[i]
        if np.isnan(atr) or atr <= 0:
            atr = 0.02 * close[i]  # Fallback to 2% of price
        
        # === STOPLOSS LOGIC (Rule #6) ===
        if position_side == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # Trailing stop: exit if price drops 2*ATR from highest
            if close[i] < highest_since_entry - 2.0 * atr:
                signals[i] = 0.0
                position_side = 0
                partial_exit_done = False
                continue
            # Hard stoploss: exit if price drops 2.5*ATR from entry
            if close[i] < entry_price - 2.5 * atr:
                signals[i] = 0.0
                position_side = 0
                partial_exit_done = False
                continue
            # Take profit: reduce to half at 2R (2 * 2.5*ATR = 5*ATR)
            if not partial_exit_done and close[i] > entry_price + 5.0 * atr:
                signals[i] = SIZE_HALF
                partial_exit_done = True
                continue
                
        elif position_side == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # Trailing stop: exit if price rises 2*ATR from lowest
            if close[i] > lowest_since_entry + 2.0 * atr:
                signals[i] = 0.0
                position_side = 0
                partial_exit_done = False
                continue
            # Hard stoploss: exit if price rises 2.5*ATR from entry
            if close[i] > entry_price + 2.5 * atr:
                signals[i] = 0.0
                position_side = 0
                partial_exit_done = False
                continue
            # Take profit: reduce to half at 2R profit
            if not partial_exit_done and close[i] < entry_price - 5.0 * atr:
                signals[i] = -SIZE_HALF
                partial_exit_done = True
                continue
        
        # === ENTRY LOGIC ===
        # Long entry: 6h uptrend + 1h uptrend + MACD bullish + RSI not overbought
        if position_side == 0 and is_uptrend_6h and is_uptrend_1h and macd_bullish and rsi_neutral:
            signals[i] = SIZE_ENTRY
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            partial_exit_done = False
            
        # Short entry: 6h downtrend + 1h downtrend + MACD bearish + RSI not oversold
        elif position_side == 0 and is_downtrend_6h and is_downtrend_1h and macd_bearish and rsi_neutral:
            signals[i] = -SIZE_ENTRY
            position_side = -1
            entry_price = close[i]
            lowest_since_entry = low[i]
            partial_exit_done = False
            
        # === EXIT LOGIC (Momentum reversal) ===
        # Exit long when MACD turns bearish
        elif position_side == 1 and macd_bearish:
            signals[i] = 0.0
            position_side = 0
            partial_exit_done = False
            
        # Exit short when MACD turns bullish
        elif position_side == -1 and macd_bullish:
            signals[i] = 0.0
            position_side = 0
            partial_exit_done = False
            
        # === TREND REVERSAL EXIT ===
        # Exit long if 6h trend reverses
        elif position_side == 1 and is_downtrend_6h:
            signals[i] = 0.0
            position_side = 0
            partial_exit_done = False
            
        # Exit short if 6h trend reverses
        elif position_side == -1 and is_uptrend_6h:
            signals[i] = 0.0
            position_side = 0
            partial_exit_done = False
            
        # Otherwise maintain current position
        else:
            signals[i] = signals[i-1] if i > 0 else 0.0
    
    return signals