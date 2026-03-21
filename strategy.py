#!/usr/bin/env python3
"""
Experiment #141: 1h Fisher Transform + 4h HMA Trend + RSI Filter
Hypothesis: 1h timeframe needs simpler logic than failed #129/#135. Fisher Transform
catches reversals in bear/range markets (2022, 2025) better than pure trend following.
4h HMA provides major trend filter without over-filtering. RSI(14) >50/<50 confirms
momentum direction. This combines reversal entry (Fisher) with trend filter (4h HMA)
for better risk-adjusted returns in both bull and bear regimes.
Position sizing: 0.25 entry, 0.12 at 2R profit, stoploss at 2.5*ATR.
Timeframe: 1h for balance between signal frequency and noise reduction.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_rsi_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    # Calculate median price
    median = (high + low) / 2
    
    # Normalize price to range -1 to +1
    hh = pd.Series(median).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(median).rolling(window=period, min_periods=period).min().values
    
    for i in range(period, n):
        hh_ll = hh[i] - ll[i]
        if hh_ll > 0:
            normalized = 2 * (median[i] - ll[i]) / hh_ll - 1
            # Apply exponential smoothing to normalized value
            if i == period:
                smooth = normalized
            else:
                smooth = 0.7 * normalized + 0.3 * smooth_prev
            
            smooth_prev = smooth
            
            # Clamp to avoid division errors
            smooth = np.clip(smooth, -0.999, 0.999)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + smooth) / (1 - smooth))
            
            # Trigger line (1-period lag of fisher)
            if i > period:
                trigger[i] = fisher[i-1]
        else:
            fisher[i] = 0.0
            trigger[i] = 0.0
    
    return fisher, trigger

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    fisher, trigger = calculate_fisher_transform(high, low, close, 9)
    hma_1h = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # 4h trend filter (major trend direction)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h HMA trend confirmation
        hma_1h_bullish = close[i] > hma_1h[i]
        hma_1h_bearish = close[i] < hma_1h[i]
        
        # Fisher Transform signals
        fisher_long = fisher[i] > -1.5 and trigger[i] <= -1.5  # cross above -1.5
        fisher_short = fisher[i] < 1.5 and trigger[i] >= 1.5   # cross below +1.5
        
        # Fisher momentum (rising/falling)
        fisher_rising = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_falling = fisher[i] < fisher[i-1] if i > 0 else False
        
        # RSI momentum confirmation
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        rsi_neutral = 45 <= rsi[i] <= 55
        
        # RSI extremes for mean reversion
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        new_signal = 0.0
        
        # LONG entries: Fisher reversal + trend filter + RSI confirmation
        # Condition 1: Fisher cross above -1.5 (reversal from oversold)
        if fisher_long and trend_bullish and rsi_bullish:
            new_signal = SIZE_ENTRY
        # Condition 2: Fisher rising from deep oversold + 4h trend bullish
        elif fisher[i] < -1.0 and fisher_rising and trend_bullish and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Condition 3: 1h HMA cross above + 4h bullish (trend continuation)
        elif hma_1h_bullish and hma_1h[i-1] <= close[i-1] and trend_bullish and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        
        # SHORT entries: Fisher reversal + trend filter + RSI confirmation
        # Condition 1: Fisher cross below +1.5 (reversal from overbought)
        if fisher_short and trend_bearish and rsi_bearish:
            new_signal = -SIZE_ENTRY
        # Condition 2: Fisher falling from deep overbought + 4h trend bearish
        elif fisher[i] > 1.0 and fisher_falling and trend_bearish and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Condition 3: 1h HMA cross below + 4h bearish (trend continuation)
        elif hma_1h_bearish and hma_1h[i-1] >= close[i-1] and trend_bearish and rsi[i] < 55:
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