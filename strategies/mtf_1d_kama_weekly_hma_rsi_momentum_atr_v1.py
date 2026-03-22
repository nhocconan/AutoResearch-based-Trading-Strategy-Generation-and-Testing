#!/usr/bin/env python3
"""
Experiment #348: 1d KAMA Trend + Weekly HMA + RSI Momentum + ATR Stop
Hypothesis: Daily timeframe with adaptive KAMA captures trends better than fixed EMA.
KAMA adjusts speed based on Efficiency Ratio - faster in trends, slower in ranges.
Weekly HMA provides macro trend bias (soft filter, not hard requirement). RSI(14)
with loose thresholds (30-70) ensures sufficient trade frequency on daily bars.
ATR(14) stoploss at 2.5x protects capital during reversals.
Timeframe: 1d (REQUIRED), HTF: 1w for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 25-50 trades total across train+test.
Key insight: KAMA adapts to volatility regime better than EMA, fewer whipsaws in ranges.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_weekly_hma_rsi_momentum_atr_v1"
timeframe = "1d"
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
        # Efficiency Ratio: net change / sum of absolute changes
        if i >= er_period:
            change = np.abs(close[i] - close[i - er_period])
            volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            er = change / volatility if volatility > 0 else 0.0
        else:
            er = 0.0
        
        # Smoothing constant
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
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Weekly macro trend bias (SOFT filter - boosts confidence, not required)
        weekly_bullish = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        weekly_bearish = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # KAMA trend (price relative to KAMA)
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # KAMA crossover signal (PRIMARY entry trigger)
        kama_cross_long = close[i] > kama[i] and close[i-1] <= kama[i-1]
        kama_cross_short = close[i] < kama[i] and close[i-1] >= kama[i-1]
        
        # RSI momentum filter (LOOSE for daily - ensure trades)
        rsi_ok_long = rsi[i] > 30  # Not deeply oversold
        rsi_ok_short = rsi[i] < 70  # Not deeply overbought
        
        # RSI confirmation
        rsi_strong_long = rsi[i] > 40
        rsi_strong_short = rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: KAMA cross + Weekly bullish + RSI ok
        if kama_cross_long and weekly_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: Price above KAMA + Weekly bullish + RSI strong
        elif kama_bullish and weekly_bullish and rsi_strong_long:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA cross without weekly filter (momentum only - ensures trades)
        elif kama_cross_long and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: KAMA cross + Weekly bearish + RSI ok
        if kama_cross_short and weekly_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Price below KAMA + Weekly bearish + RSI strong
        elif kama_bearish and weekly_bearish and rsi_strong_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA cross without weekly filter (momentum only - ensures trades)
        elif kama_cross_short and rsi[i] < 65:
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