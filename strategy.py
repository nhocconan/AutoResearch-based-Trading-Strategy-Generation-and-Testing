#!/usr/bin/env python3
"""
Experiment #087: 1h KAMA Adaptive Trend + 4h HMA Filter + RSI Pullback
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility - faster in trends,
slower in ranges. This should outperform fixed EMA in mixed bull/bear markets (2021-2026).
Combine with 4h HMA trend filter (proven from winning strategies) and RSI pullback entries
(proven from mtf_12h_supertrend_daily_hma_rsi_pullback_v2 which has Sharpe=0.499).
Key difference from failures: Use adaptive KAMA instead of fixed EMA/Supertrend, keep
RSI pullback logic (proven), use 4h HMA (not daily - more responsive for 1h entries).
Position sizing: 0.25 entry, 0.15 at 1.5R profit, stoploss at 2.5*ATR trailing.
Timeframe: 1h (required for this experiment) with 4h HTF reference.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_hma_rsi_pullback_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility - moves fast in trends, slow in ranges.
    Efficiency Ratio (ER) determines smoothing constant.
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    
    # Change = absolute difference from period ago
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    # Volatility = sum of absolute price changes over period
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Dynamic smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # KAMA for adaptive trend following
    kama_fast = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, period=20, fast_period=2, slow_period=30)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # 4h trend filter (HTF) - price relative to 4h HMA
        hma_4h_val = hma_4h_aligned[i]
        if np.isnan(hma_4h_val) or hma_4h_val == 0:
            hma_4h_val = close[i]  # fallback if alignment fails
        
        trend_bullish = close[i] > hma_4h_val
        trend_bearish = close[i] < hma_4h_val
        
        # KAMA crossover signals
        kama_cross_long = kama_fast[i] > kama_slow[i] and (i > 0 and kama_fast[i-1] <= kama_slow[i-1])
        kama_cross_short = kama_fast[i] < kama_slow[i] and (i > 0 and kama_fast[i-1] >= kama_slow[i-1])
        
        # KAMA trend state
        kama_trend_long = kama_fast[i] > kama_slow[i]
        kama_trend_short = kama_fast[i] < kama_slow[i]
        
        # KAMA slope confirmation
        kama_slope_long = kama_fast[i] > kama_fast[i-1] if i > 0 else False
        kama_slope_short = kama_fast[i] < kama_fast[i-1] if i > 0 else False
        
        # RSI pullback conditions (proven from winning strategy)
        # Long: RSI dipped to 35-50 zone in uptrend (pullback entry)
        rsi_pullback_long = 35 < rsi[i] < 55
        # Short: RSI rallied to 45-65 zone in downtrend (pullback entry)
        rsi_pullback_short = 45 < rsi[i] < 65
        
        # RSI extreme filter (avoid entering at extremes)
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        new_signal = 0.0
        
        # LONG ENTRY conditions
        # Condition 1: KAMA cross + 4h bullish + RSI pullback (primary entry)
        if kama_cross_long and trend_bullish and rsi_pullback_long and rsi_not_overbought:
            new_signal = SIZE_ENTRY
        # Condition 2: KAMA trend + 4h bullish + RSI not extreme (continuation)
        elif kama_trend_long and trend_bullish and kama_slope_long and rsi_not_overbought and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Condition 3: Price above both KAMA + 4h bullish + RSI rising
        elif close[i] > kama_fast[i] and close[i] > kama_slow[i] and trend_bullish and rsi[i] > rsi[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Condition 1: KAMA cross + 4h bearish + RSI pullback (primary entry)
        if kama_cross_short and trend_bearish and rsi_pullback_short and rsi_not_oversold:
            new_signal = -SIZE_ENTRY
        # Condition 2: KAMA trend + 4h bearish + RSI not extreme (continuation)
        elif kama_trend_short and trend_bearish and kama_slope_short and rsi_not_oversold and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Condition 3: Price below both KAMA + 4h bearish + RSI falling
        elif close[i] < kama_fast[i] and close[i] < kama_slow[i] and trend_bearish and rsi[i] < rsi[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
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
            elif not position_reduced and entry_atr > 0:
                # Take profit at 1.5R
                profit = close[i] - entry_price
                risk = 2.5 * entry_atr
                if profit >= 1.5 * risk:
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
            elif not position_reduced and entry_atr > 0:
                # Take profit at 1.5R
                profit = entry_price - close[i]
                risk = 2.5 * entry_atr
                if profit >= 1.5 * risk:
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
            entry_atr = atr[i]
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
            entry_atr = atr[i]
        
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
            entry_atr = 0.0
        
        signals[i] = new_signal
    
    return signals