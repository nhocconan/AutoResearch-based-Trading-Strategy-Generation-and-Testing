#!/usr/bin/env python3
"""
Experiment #350: 30m KAMA Trend + 4h HMA Filter + RSI(7) Momentum + ATR Stop
Hypothesis: 30m timeframe offers balance between noise (5m/15m) and lag (4h/1d).
KAMA adapts to volatility - faster in trends, slower in ranges. 4h HMA provides
macro trend bias without over-filtering. RSI(7) is faster than RSI(14) for 30m
entries. Loose RSI thresholds (35-65) ensure sufficient trade frequency.
ATR(14) stoploss at 2.0x protects capital. Position size 0.25 for risk control.
Key insight: Previous KAMA strategies showed promise but were over-filtered.
This version simplifies entry logic to ensure 10+ trades while maintaining edge.
Timeframe: 30m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 30-80 trades total, DD < -30%.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_4h_hma_rsi7_momentum_atr_v1"
timeframe = "30m"
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
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if i >= er_period:
            change = np.abs(close[i] - close[i - er_period])
            volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            er = change / volatility if volatility > 0 else 0.0
        else:
            er = 0.0
        
        sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=7):
    """Calculate RSI indicator with configurable period."""
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
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    rsi = calculate_rsi(close, 7)  # Faster RSI for 30m
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_price = 0.0
    lowest_price = 0.0
    initial_risk = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias (SOFT filter - boosts confidence)
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # KAMA crossover signal (PRIMARY entry trigger)
        kama_cross_long = close[i] > kama[i] and close[i-1] <= kama[i-1]
        kama_cross_short = close[i] < kama[i] and close[i-1] >= kama[i-1]
        
        # RSI momentum filter (LOOSE for 30m - ensure trades)
        rsi_ok_long = rsi[i] > 35
        rsi_ok_short = rsi[i] < 65
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: KAMA cross + 4h bullish + RSI ok
        if kama_cross_long and trend_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: KAMA cross without 4h filter (momentum only - ensures trades)
        elif kama_cross_long and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Tertiary: Price above KAMA + strong RSI (continuation)
        elif kama_bullish and rsi[i] > 50 and rsi[i-1] <= 50:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: KAMA cross + 4h bearish + RSI ok
        if kama_cross_short and trend_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: KAMA cross without 4h filter (momentum only - ensures trades)
        elif kama_cross_short and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Tertiary: Price below KAMA + weak RSI (continuation)
        elif kama_bearish and rsi[i] < 50 and rsi[i-1] >= 50:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest price for trailing
            if close[i] > highest_price:
                highest_price = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
            current_stop = highest_price - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                profit = close[i] - entry_price
                if profit >= 2.0 * initial_risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest price for trailing
            if lowest_price == 0.0 or close[i] < lowest_price:
                lowest_price = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest)
            current_stop = lowest_price + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                profit = entry_price - close[i]
                if profit >= 2.0 * initial_risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened (from flat)
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            initial_risk = 2.0 * atr[i]
            trailing_stop = close[i] - initial_risk if position_side > 0 else close[i] + initial_risk
            highest_price = close[i] if position_side > 0 else 0.0
            lowest_price = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed (long to short or short to long)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            initial_risk = 2.0 * atr[i]
            trailing_stop = close[i] - initial_risk if position_side > 0 else close[i] + initial_risk
            highest_price = close[i] if position_side > 0 else 0.0
            lowest_price = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit partial)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_price = 0.0
            lowest_price = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals