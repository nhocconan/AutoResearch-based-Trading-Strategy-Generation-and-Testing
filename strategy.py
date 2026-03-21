#!/usr/bin/env python3
"""
Experiment #052: 4h Supertrend with Daily HMA Trend Filter + RSI Momentum
Hypothesis: 4h timeframe offers good balance between trade frequency and noise reduction.
Use 1d HMA(50) as primary trend filter (bullish when price > HMA, bearish when < HMA).
Enter on 4h Supertrend flips aligned with daily trend direction.
Add RSI momentum filter (RSI > 55 for longs, RSI < 45 for shorts) to avoid weak signals.
Use 2.5*ATR trailing stop for wider stops on 4h timeframe (reduces premature exits).
Position sizing: 0.30 entry, reduce to 0.15 at 2R profit, exit at stoploss.
Key difference from #046: Simpler entry logic, fewer conflicting filters, wider stops.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_daily_hma_rsi_momentum_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower[0]
    direction[0] = 1
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # 4h HMA for local trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    entry_atr = 0.0
    
    for i in range(100, n):
        # Daily trend filter (HTF) - price relative to 1d HMA(50)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 4h Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip signals
        st_flip_long = (i > 0) and (st_direction[i] == 1) and (st_direction[i-1] == -1)
        st_flip_short = (i > 0) and (st_direction[i] == -1) and (st_direction[i-1] == 1)
        
        # 4h HMA trend confirmation
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # RSI momentum filter (avoid weak signals)
        rsi_momentum_long = rsi[i] > 55  # Bullish momentum
        rsi_momentum_short = rsi[i] < 45  # Bearish momentum
        
        # RSI not overextended (avoid chasing tops/bottoms)
        rsi_not_overbought = rsi[i] < 75
        rsi_not_oversold = rsi[i] > 25
        
        new_signal = 0.0
        
        # LONG ENTRY: Supertrend flip + Daily bullish + RSI momentum
        if st_flip_long and daily_bullish and rsi_momentum_long and rsi_not_overbought:
            new_signal = SIZE_ENTRY
        # Alternative: Supertrend long + Daily bullish + HMA trend + RSI ok
        elif st_long and daily_bullish and hma_trend_long and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: Supertrend flip + Daily bearish + RSI momentum
        if st_flip_short and daily_bearish and rsi_momentum_short and rsi_not_oversold:
            new_signal = -SIZE_ENTRY
        # Alternative: Supertrend short + Daily bearish + HMA trend + RSI ok
        elif st_short and daily_bearish and hma_trend_short and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Calculate trailing stop (wider for 4h: 2.5*ATR)
            current_stop = close[i] - 2.5 * entry_atr
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = close[i] - entry_price
                    risk = entry_atr * 2.5
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = SIZE_HALF
                        position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Calculate trailing stop (wider for 4h: 2.5*ATR)
            current_stop = close[i] + 2.5 * entry_atr
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = entry_price - close[i]
                    risk = entry_atr * 2.5
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = -SIZE_HALF
                        position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            entry_atr = atr[i]
            trailing_stop = close[i] - 2.5 * entry_atr if position_side > 0 else close[i] + 2.5 * entry_atr
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            entry_atr = atr[i]
            trailing_stop = close[i] - 2.5 * entry_atr if position_side > 0 else close[i] + 2.5 * entry_atr
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            position_reduced = False
            entry_atr = 0.0
        
        signals[i] = new_signal
    
    return signals