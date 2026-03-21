#!/usr/bin/env python3
"""
Experiment #119: 12h Donchian Breakout with Daily HMA Trend + RSI Filter
Hypothesis: Current best uses Supertrend flips. Try Donchian breakouts instead,
which catch trends earlier at range boundaries. Keep Daily HMA trend filter
(proven to work). Add RSI confirmation (not extreme) to avoid false breakouts.
Simplify entry logic to ensure 10+ trades (learning from 0-trade failures).
12h timeframe reduces noise vs lower TFs while maintaining trade frequency.
Position sizing: 0.25 entry, stoploss at 2.5*ATR trailing.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_daily_hma_rsi_v1"
timeframe = "12h"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend filter (HTF) - price relative to Daily HMA
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # RSI filter (avoid extreme overbought/oversold for entries)
        rsi_ok_long = rsi[i] < 70  # Not extremely overbought
        rsi_ok_short = rsi[i] > 30  # Not extremely oversold
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 50  # Bullish momentum
        rsi_momentum_short = rsi[i] < 50  # Bearish momentum
        
        new_signal = 0.0
        
        # LONG ENTRY: Daily bullish + Donchian breakout + RSI confirmation
        if daily_bullish and breakout_long and rsi_ok_long and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: Daily bearish + Donchian breakout + RSI confirmation
        elif daily_bearish and breakout_short and rsi_ok_short and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest price for trailing
            if close[i] > highest_price:
                highest_price = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_price - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            # Update lowest price for trailing
            if lowest_price == 0.0 or close[i] < lowest_price:
                lowest_price = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_price + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            if position_side > 0:
                trailing_stop = close[i] - 2.5 * atr[i]
                highest_price = close[i]
                lowest_price = 0.0
            else:
                trailing_stop = close[i] + 2.5 * atr[i]
                lowest_price = close[i]
                highest_price = 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            if position_side > 0:
                trailing_stop = close[i] - 2.5 * atr[i]
                highest_price = close[i]
                lowest_price = 0.0
            else:
                trailing_stop = close[i] + 2.5 * atr[i]
                lowest_price = close[i]
                highest_price = 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_price = 0.0
            lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals