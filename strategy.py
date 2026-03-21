#!/usr/bin/env python3
"""
Experiment #261: 1h KAMA Adaptive Trend + 4h HMA Filter with Simple Momentum
Hypothesis: Previous strategies failed due to too many restrictive filters (0 trades).
This uses SIMPLER entry conditions: 4h HMA for trend bias, KAMA adaptive MA for 
entries, MACD histogram for momentum, loose RSI filter (>45/<55 not extremes).
Key insight: OR conditions instead of AND, fewer filters = more trades while 
maintaining trend alignment. Position sizing: 0.28 entry, 0.14 half at 2R.
Stoploss: 2.5*ATR trailing. Target: Beat Sharpe=0.499 with more consistent trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_hma_macd_simple_momentum_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    change = np.abs(close_s.diff())
    volatility = pd.Series(close).diff(er_period).abs()
    er = change.rolling(er_period, min_periods=er_period).sum() / volatility.replace(0, np.nan)
    er = er.fillna(0)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = pd.Series(index=close_s.index, dtype=float)
    kama.iloc[er_period] = close_s.iloc[er_period]
    for i in range(er_period + 1, len(close_s)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama.iloc[i-1])
    return kama.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend filter
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align 4h HMA to 1h (Rule 2 - auto shift for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    rsi = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # Track previous values for momentum detection
    prev_macd_hist = np.roll(macd_hist, 1)
    prev_macd_hist[0] = macd_hist[0]
    prev_kama = np.roll(kama, 1)
    prev_kama[0] = kama[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(200, n):
        # 4h trend filter (SIMPLE - just price vs HMA)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # KAMA adaptive trend
        kama_bullish = close[i] > kama[i] and kama[i] > prev_kama[i]
        kama_bearish = close[i] < kama[i] and kama[i] < prev_kama[i]
        
        # MACD momentum (loose filter)
        macd_positive = macd_hist[i] > 0
        macd_negative = macd_hist[i] < 0
        macd_rising = macd_hist[i] > prev_macd_hist[i]
        macd_falling = macd_hist[i] < prev_macd_hist[i]
        
        # RSI filter (LOOSE - not extremes to ensure trades)
        rsi_bullish = rsi[i] > 45
        rsi_bearish = rsi[i] < 55
        
        # SMA 200 for macro trend
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY (OR conditions for more trades) ===
        # Condition 1: KAMA cross up + 4h trend + RSI
        if kama_bullish and trend_bullish and rsi_bullish:
            new_signal = SIZE_ENTRY
        # Condition 2: MACD positive + rising + above SMA200
        elif macd_positive and macd_rising and above_sma200:
            new_signal = SIZE_ENTRY
        # Condition 3: 4h trend + KAMA above + MACD positive
        elif trend_bullish and close[i] > kama[i] and macd_positive:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY (OR conditions for more trades) ===
        # Condition 1: KAMA cross down + 4h trend + RSI
        if kama_bearish and trend_bearish and rsi_bearish:
            new_signal = -SIZE_ENTRY
        # Condition 2: MACD negative + falling + below SMA200
        elif macd_negative and macd_falling and below_sma200:
            new_signal = -SIZE_ENTRY
        # Condition 3: 4h trend + KAMA below + MACD negative
        elif trend_bearish and close[i] < kama[i] and macd_negative:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals