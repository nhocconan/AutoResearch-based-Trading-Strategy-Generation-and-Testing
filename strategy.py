#!/usr/bin/env python3
"""
Experiment #098: 30m Volatility Breakout with 4h HMA Trend + RSI Momentum
Hypothesis: Previous 30m strategies failed due to over-filtering (CRSI+Chop, Donchian+RSI).
Try simpler volatility breakout (Larry Williams style) with HTF trend filter.
Entry: Price breaks ATR-based volatility level + 4h HMA confirms direction + RSI momentum.
This differs from failed #092 (Donchian) by using ATR-based dynamic levels instead of fixed lookback.
Position sizing: 0.25 entry, 0.125 at 1.5R profit, stoploss at 2*ATR trailing.
30m timeframe captures intraday moves while 4h HMA filters noise.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_volbreakout_4h_hma_rsi_v1"
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

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

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
    
    # Volatility breakout levels (Larry Williams style)
    # Long breakout: high > prev_high + 0.5*ATR
    # Short breakout: low < prev_low - 0.5*ATR
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Volatility breakout thresholds
    long_breakout_level = prev_high + 0.5 * atr
    short_breakout_level = prev_low - 0.5 * atr
    
    # SMA for trend confirmation
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # 4h trend filter (HTF)
        hma_4h_val = hma_4h_aligned[i]
        hma_valid = not np.isnan(hma_4h_val) and hma_4h_val > 0
        
        if hma_valid:
            hma_bullish = close[i] > hma_4h_val
            hma_bearish = close[i] < hma_4h_val
        else:
            hma_bullish = False
            hma_bearish = False
        
        # 30m trend confirmation
        sma_valid = not np.isnan(sma_50[i]) and not np.isnan(sma_200[i])
        if sma_valid:
            sma_bullish = close[i] > sma_50[i] and sma_50[i] > sma_200[i]
            sma_bearish = close[i] < sma_50[i] and sma_50[i] < sma_200[i]
        else:
            sma_bullish = False
            sma_bearish = False
        
        # RSI momentum filter (not too strict)
        rsi_bullish = rsi[i] > 45 and rsi[i] < 75
        rsi_bearish = rsi[i] < 55 and rsi[i] > 25
        
        # Volatility breakout signals
        long_breakout = high[i] > long_breakout_level[i]
        short_breakout = low[i] < short_breakout_level[i]
        
        new_signal = 0.0
        
        # LONG ENTRY: breakout + HTF bullish + RSI momentum
        if long_breakout and hma_bullish and rsi_bullish:
            new_signal = SIZE_ENTRY
        # LONG ENTRY: breakout + SMA trend + RSI momentum (fallback)
        elif long_breakout and sma_bullish and rsi_bullish:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: breakout + HTF bearish + RSI momentum
        if short_breakout and hma_bearish and rsi_bearish:
            new_signal = -SIZE_ENTRY
        # SHORT ENTRY: breakout + SMA trend + RSI momentum (fallback)
        elif short_breakout and sma_bearish and rsi_bearish:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                risk = 2.0 * entry_atr
                profit = close[i] - entry_price
                if profit >= 1.5 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                risk = 2.0 * entry_atr
                profit = entry_price - close[i]
                if profit >= 1.5 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            entry_atr = atr[i]
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            entry_atr = atr[i]
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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
            entry_atr = 0.0
        
        signals[i] = new_signal
    
    return signals