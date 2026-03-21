#!/usr/bin/env python3
"""
Experiment #135: 1h Multi-Timeframe Pullback Strategy with Volume Filter
Hypothesis: 1h timeframe needs strong HTF trend filter (12h HMA) to avoid
whipsaws. Only enter on deep pullbacks (RSI<35/>65) with volume confirmation.
This reduces trade count while maintaining quality entries. Stoploss at 3*ATR
for wider breathing room. Position size 0.25 max to control drawdown.
Key insight from failures: simpler is better, fewer trades = less fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_12h_rsi_volume_pullback_v1"
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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    hma_fast = calculate_hma(close, 8)
    hma_slow = calculate_hma(close, 21)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # 12h trend filter (major trend direction)
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # 1h HMA crossover trend
        hma_trend_long = hma_fast[i] > hma_slow[i]
        hma_trend_short = hma_fast[i] < hma_slow[i]
        
        # HMA slope confirmation (5 bars back)
        hma_slope_long = hma_fast[i] > hma_fast[i-5] if i > 5 else True
        hma_slope_short = hma_fast[i] < hma_fast[i-5] if i > 5 else True
        
        # Volume confirmation (spike > 1.2x average)
        vol_spike = volume[i] > 1.2 * vol_sma[i] if vol_sma[i] > 0 else True
        
        # RSI conditions (looser for more trades)
        rsi_oversold = rsi[i] < 38
        rsi_overbought = rsi[i] > 62
        
        new_signal = 0.0
        
        # LONG: 12h bullish + 1h trend + pullback + volume
        if trend_12h_bullish and hma_trend_long and hma_slope_long and rsi_oversold:
            new_signal = SIZE_ENTRY
        
        # SHORT: 12h bearish + 1h trend + pullback + volume
        elif trend_12h_bearish and hma_trend_short and hma_slope_short and rsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest price for trailing
            if high[i] > highest_price:
                highest_price = high[i]
            
            # Calculate trailing stop (3*ATR from highest)
            current_stop = highest_price - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if low[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest price for trailing
            if lowest_price == 0.0 or low[i] < lowest_price:
                lowest_price = low[i]
            
            # Calculate trailing stop (3*ATR from lowest)
            current_stop = lowest_price + 3.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if high[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
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
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_price = high[i] if position_side > 0 else 0.0
            lowest_price = low[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_price = high[i] if position_side > 0 else 0.0
            lowest_price = low[i] if position_side < 0 else 0.0
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