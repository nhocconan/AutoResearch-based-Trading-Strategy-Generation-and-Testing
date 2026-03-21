#!/usr/bin/env python3
"""
Experiment #320: 30m EMA Crossover + 4h HMA Trend + RSI Momentum + ATR Stops
Hypothesis: 30m EMA crossover (8/21) provides timely entry signals while 4h HMA(21) 
gives reliable trend bias. RSI(14) > 50 confirms momentum for longs, < 50 for shorts.
ATR(14) trailing stops at 2.5x protect capital. Simpler than regime-based approaches
that failed repeatedly. Target: Generate 50-100 trades/year with Sharpe > 0.5.
Timeframe: 30m (required), HTF: 4h for trend bias.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_ema_crossover_4h_hma_rsi_momentum_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Track previous values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_ema_fast = np.roll(ema_fast, 1)
    prev_ema_fast[0] = ema_fast[0]
    prev_ema_slow = np.roll(ema_slow, 1)
    prev_ema_slow[0] = ema_slow[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias
        hma_valid = not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_valid and close[i] < hma_4h_aligned[i]
        
        # EMA crossover signals
        ema_cross_long = ema_fast[i] > ema_slow[i] and prev_ema_fast[i] <= prev_ema_slow[i]
        ema_cross_short = ema_fast[i] < ema_slow[i] and prev_ema_fast[i] >= prev_ema_slow[i]
        
        # EMA alignment (fast > slow > 50 for strong trend)
        ema_aligned_long = ema_fast[i] > ema_slow[i] and ema_slow[i] > ema_50[i]
        ema_aligned_short = ema_fast[i] < ema_slow[i] and ema_slow[i] < ema_50[i]
        
        # RSI momentum
        rsi_bullish = rsi[i] > 50 and prev_rsi[i] <= 50
        rsi_bearish = rsi[i] < 50 and prev_rsi[i] >= 50
        rsi_momentum_long = rsi[i] > 50
        rsi_momentum_short = rsi[i] < 50
        
        # Price above/below 4h HMA
        price_above_hma = hma_valid and close[i] > hma_4h_aligned[i]
        price_below_hma = hma_valid and close[i] < hma_4h_aligned[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: 4h bullish + EMA crossover + RSI momentum
        if trend_bullish and ema_cross_long and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Secondary: 4h bullish + EMA aligned + RSI > 50
        elif trend_bullish and ema_aligned_long and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Tertiary: EMA crossover + RSI crosses 50 + price above 4h HMA
        elif ema_cross_long and rsi_bullish and price_above_hma:
            new_signal = SIZE_ENTRY
        # Quaternary: Simple trend following (EMA aligned + 4h bias)
        elif ema_aligned_long and trend_bullish and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: 4h bearish + EMA crossover + RSI momentum
        if trend_bearish and ema_cross_short and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Secondary: 4h bearish + EMA aligned + RSI < 50
        elif trend_bearish and ema_aligned_short and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: EMA crossover + RSI crosses 50 + price below 4h HMA
        elif ema_cross_short and rsi_bearish and price_below_hma:
            new_signal = -SIZE_ENTRY
        # Quaternary: Simple trend following (EMA aligned + 4h bias)
        elif ema_aligned_short and trend_bearish and rsi[i] < 55:
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