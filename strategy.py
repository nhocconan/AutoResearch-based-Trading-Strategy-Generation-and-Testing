#!/usr/bin/env python3
"""
Experiment #286: 4h HMA Trend with Daily Regime Filter and Donchian Breakout
Hypothesis: 4h timeframe captures medium-term trends while avoiding 2022 whipsaw.
Daily HMA provides regime filter (only trade with daily trend). Donchian(20) breakout
captures momentum moves. RSI filter avoids extreme entries. Simple logic = more trades.
Position sizing: 0.28 entry, 0.14 half at 2R. Stoploss: 2.5*ATR trailing.
Key insight from failures: fewer filters = more trades. Looser RSI (25-75) ensures entries.
Target: Beat Sharpe=0.499 from mtf_12h_supertrend_daily_hma_rsi_pullback_v2
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_daily_regime_donchian_rsi_atr_v1"
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
    """Calculate Donchian Channel upper and lower bands."""
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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    hma_4h = calculate_hma(close, 21)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Track previous values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    prev_hma_4h = np.roll(hma_4h, 1)
    prev_hma_4h[0] = hma_4h[0]
    
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
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # Daily regime filter (simple - only trade with daily trend)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 4h HMA trend
        hma_4h_bullish = hma_4h[i] > prev_hma_4h[i] and close[i] > hma_4h[i]
        hma_4h_bearish = hma_4h[i] < prev_hma_4h[i] and close[i] < hma_4h[i]
        
        # Donchian breakout signals
        donchian_breakout_long = close[i] > donchian_upper[i] and prev_close[i] <= donchian_upper[i]
        donchian_breakout_short = close[i] < donchian_lower[i] and prev_close[i] >= donchian_lower[i]
        
        # RSI filter (loose thresholds to ensure trades)
        rsi_ok_long = rsi[i] < 75  # avoid overbought
        rsi_ok_short = rsi[i] > 25  # avoid oversold
        rsi_momentum_long = rsi[i] > 50 and prev_rsi[i] <= 50
        rsi_momentum_short = rsi[i] < 50 and prev_rsi[i] >= 50
        
        # HMA crossover
        hma_cross_up = prev_hma_4h[i] >= close[i] and hma_4h[i] < close[i]
        hma_cross_down = prev_hma_4h[i] <= close[i] and hma_4h[i] > close[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Donchian breakout with daily trend + RSI filter
        if donchian_breakout_long and daily_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        
        # HMA crossover with daily trend
        elif hma_cross_up and daily_bullish and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        
        # 4h trend continuation with RSI momentum
        elif hma_4h_bullish and daily_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Donchian breakout with daily trend + RSI filter
        if donchian_breakout_short and daily_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        
        # HMA crossover with daily trend
        elif hma_cross_down and daily_bearish and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        
        # 4h trend continuation with RSI momentum
        elif hma_4h_bearish and daily_bearish and rsi_momentum_short:
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