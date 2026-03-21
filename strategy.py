#!/usr/bin/env python3
"""
Experiment #256: 4h Trend Momentum with Daily/Weekly HMA Filter
Hypothesis: 4h timeframe captures medium-term trends better than 12h for entry timing.
Using 1d HMA(21) as primary trend filter, 1w HMA(21) as macro confirmation. Entry on
MACD histogram momentum shift + price above/below HMA. Simpler logic than previous
attempts to ensure sufficient trades. ATR(14) trailing stop at 2.5*ATR. Position sizing:
0.30 entry, 0.15 at 2R profit. Key difference from failures: fewer conflicting filters,
OR logic for entry conditions (not AND), ensuring trades generate in both bull/bear.
Target: Beat Sharpe=0.499 from current best (12h supertrend).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_macd_momentum_daily_weekly_hma_atr_v1"
timeframe = "4h"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    
    # Track previous values for momentum detection
    prev_macd_hist = np.roll(macd_hist, 1)
    prev_macd_hist[0] = macd_hist[0]
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    ATR_MULT = 2.5
    
    # Track positions for stoploss/takeprofit
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # HTF trend filters (use OR logic to ensure trades)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # MACD momentum signals
        macd_increasing = macd_hist[i] > prev_macd_hist[i]
        macd_decreasing = macd_hist[i] < prev_macd_hist[i]
        macd_positive = macd_hist[i] > 0
        macd_negative = macd_hist[i] < 0
        macd_cross_up = prev_macd_hist[i] <= 0 and macd_hist[i] > 0
        macd_cross_down = prev_macd_hist[i] >= 0 and macd_hist[i] < 0
        
        # RSI momentum (loose filters to ensure trades)
        rsi_bullish = rsi[i] > 45
        rsi_bearish = rsi[i] < 55
        rsi_strong_bull = rsi[i] > 50
        rsi_strong_bear = rsi[i] < 50
        
        # Price momentum
        price_up = close[i] > prev_close[i]
        price_down = close[i] < prev_close[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY (multiple paths to ensure trades) ===
        # Path 1: MACD cross up with daily trend
        if macd_cross_up and daily_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 2: MACD positive + increasing + weekly trend
        elif macd_positive and macd_increasing and weekly_bullish:
            if rsi_bullish or price_up:
                new_signal = SIZE_ENTRY
        
        # Path 3: Price above both HMA + MACD positive
        elif daily_bullish and weekly_bullish and macd_positive:
            if price_up:
                new_signal = SIZE_ENTRY
        
        # Path 4: Strong momentum with daily trend
        elif macd_increasing and rsi_strong_bull and daily_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY (multiple paths to ensure trades) ===
        # Path 1: MACD cross down with daily trend
        if macd_cross_down and daily_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 2: MACD negative + decreasing + weekly trend
        elif macd_negative and macd_decreasing and weekly_bearish:
            if rsi_bearish or price_down:
                new_signal = -SIZE_ENTRY
        
        # Path 3: Price below both HMA + MACD negative
        elif daily_bearish and weekly_bearish and macd_negative:
            if price_down:
                new_signal = -SIZE_ENTRY
        
        # Path 4: Strong momentum with daily trend
        elif macd_decreasing and rsi_strong_bear and daily_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest price for trailing
            if high[i] > highest_price:
                highest_price = high[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_price - ATR_MULT * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if low[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = ATR_MULT * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest price for trailing
            if low[i] < lowest_price or lowest_price == 0.0:
                lowest_price = low[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_price + ATR_MULT * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if high[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = ATR_MULT * atr[i]
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
            if position_side > 0:
                trailing_stop = close[i] - ATR_MULT * atr[i]
                highest_price = high[i]
                lowest_price = 0.0
            else:
                trailing_stop = close[i] + ATR_MULT * atr[i]
                highest_price = 0.0
                lowest_price = low[i]
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            if position_side > 0:
                trailing_stop = close[i] - ATR_MULT * atr[i]
                highest_price = high[i]
                lowest_price = 0.0
            else:
                trailing_stop = close[i] + ATR_MULT * atr[i]
                highest_price = 0.0
                lowest_price = low[i]
            position_reduced = False
        
        # Position reduced (take profit)
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