#!/usr/bin/env python3
"""
Experiment #105: 1h RSI Pullback with 4h HMA Trend + Volatility Filter
Hypothesis: Recent failures show complex regime-adaptive strategies overfit. 
Return to proven components: 4h HMA trend filter + RSI pullback entries.
Add volatility filter to avoid entering during extreme moves (reduces whipsaw).
Use simpler entry logic to ensure 10+ trades per symbol (learning from 0-trade failures).
1h timeframe should balance trade frequency vs noise better than 15m/30m failures.
Position sizing: 0.25 entry, 0.125 at 2R profit, stoploss at 2.5*ATR trailing.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_4h_hma_vol_filter_v1"
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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # Volatility filter: ATR percentile
    atr_sma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / np.maximum(atr_sma, 1e-10)
    
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
    
    for i in range(250, n):
        # 4h trend filter (HTF)
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        if not hma_4h_valid:
            signals[i] = 0.0
            continue
            
        daily_bullish = close[i] > hma_4h_aligned[i]
        daily_bearish = close[i] < hma_4h_aligned[i]
        
        # SMA200 filter for additional trend confirmation
        sma_200_valid = not np.isnan(sma_200[i])
        if sma_200_valid:
            above_sma200 = close[i] > sma_200[i]
            below_sma200 = close[i] < sma_200[i]
        else:
            above_sma200 = True
            below_sma200 = True
        
        # Volatility filter: avoid extreme volatility (ratio > 2.0 or < 0.5)
        vol_normal = 0.5 < atr_ratio[i] < 2.0 if not np.isnan(atr_ratio[i]) else True
        
        # RSI pullback signals (simpler thresholds to ensure trades)
        rsi_oversold = rsi[i] < 45  # Buy pullback in uptrend
        rsi_overbought = rsi[i] > 55  # Sell pullback in downtrend
        
        # RSI momentum confirmation
        rsi_rising = i > 0 and rsi[i] > rsi[i-1]
        rsi_falling = i > 0 and rsi[i] < rsi[i-1]
        
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + RSI pullback + normal vol
        if daily_bullish and above_sma200 and rsi_oversold and vol_normal:
            if rsi_rising or (i > 1 and rsi[i] > rsi[i-2]):
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: 4h bearish + RSI pullback + normal vol
        if daily_bearish and below_sma200 and rsi_overbought and vol_normal:
            if rsi_falling or (i > 1 and rsi[i] < rsi[i-2]):
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
                profit = close[i] - entry_price
                risk = 2.5 * entry_atr
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
                profit = entry_price - close[i]
                risk = 2.5 * entry_atr
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