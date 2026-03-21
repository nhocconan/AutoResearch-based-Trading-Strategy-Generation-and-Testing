#!/usr/bin/env python3
"""
Experiment #144: 1d Supertrend + 4h HMA Trend Filter + RSI Pullback
Hypothesis: Daily Supertrend provides clear trend signals with built-in ATR stop.
4h HMA acts as major trend filter (bullish when price > 4h HMA). RSI pullback 
entries improve timing and reduce false breakouts. This simplifies the complex 
regime detection from #132 that blocked too many signals. Focus on fewer but 
higher quality trades with proper MTF alignment.
Position sizing: 0.25 entry, reduce to 0.125 at 2R profit, stoploss at 2.5*ATR.
Timeframe: 1d with 4h HTF reference for trend filter.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_supertrend_4h_hma_rsi_pullback_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_line, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize arrays
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    # First value
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if direction[i-1] == 1:
            # Previous trend was up
            if close[i] > supertrend[i-1]:
                # Continue uptrend
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                direction[i] = 1
            else:
                # Trend reversal to down
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Previous trend was down
            if close[i] < supertrend[i-1]:
                # Continue downtrend
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                direction[i] = -1
            else:
                # Trend reversal to up
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction

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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend_line, supertrend_dir = calculate_supertrend(high, low, close, 10, 3.0)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss and take profit
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):  # Start after 200-day SMA is valid
        # 4h HMA trend filter (major trend direction)
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 200-day SMA filter (long-term trend)
        trend_sma_bullish = close[i] > sma_200[i]
        trend_sma_bearish = close[i] < sma_200[i]
        
        # Supertrend signals
        supertrend_long = supertrend_dir[i] == 1
        supertrend_short = supertrend_dir[i] == -1
        
        # Supertrend flip detection (entry trigger)
        supertrend_flip_long = supertrend_long and supertrend_dir[i-1] == -1
        supertrend_flip_short = supertrend_short and supertrend_dir[i-1] == 1
        
        # RSI pullback conditions (better entry timing)
        rsi_pullback_long = 35 <= rsi[i] <= 55  # RSI pulled back in uptrend
        rsi_pullback_short = 45 <= rsi[i] <= 65  # RSI pulled back in downtrend
        rsi_extreme_long = rsi[i] < 40  # Oversold bounce
        rsi_extreme_short = rsi[i] > 60  # Overbought fade
        
        new_signal = 0.0
        
        # LONG ENTRY: Supertrend flip + 4h HMA bullish + RSI confirmation
        if supertrend_flip_long:
            # Strong signal: 4h HMA bullish + SMA200 bullish
            if trend_4h_bullish and trend_sma_bullish:
                new_signal = SIZE_ENTRY
            # Moderate signal: 4h HMA bullish only (with RSI pullback)
            elif trend_4h_bullish and rsi_pullback_long:
                new_signal = SIZE_ENTRY
            # Weaker signal: 4h HMA bullish + RSI extreme
            elif trend_4h_bullish and rsi_extreme_long:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: Supertrend flip + 4h HMA bearish + RSI confirmation
        elif supertrend_flip_short:
            # Strong signal: 4h HMA bearish + SMA200 bearish
            if trend_4h_bearish and trend_sma_bearish:
                new_signal = -SIZE_ENTRY
            # Moderate signal: 4h HMA bearish only (with RSI pullback)
            elif trend_4h_bearish and rsi_pullback_short:
                new_signal = -SIZE_ENTRY
            # Weaker signal: 4h HMA bearish + RSI extreme
            elif trend_4h_bearish and rsi_extreme_short:
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