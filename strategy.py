#!/usr/bin/env python3
"""
Experiment #121: 15m Supertrend + 4h HMA Trend + RSI Pullback
Hypothesis: 15m timeframe captures more intraday moves than 12h/4h strategies.
Use 4h HMA for trend bias (proven in current best), 15m Supertrend for entries,
RSI pullback (40-60 range) for better entry timing. This should generate more
trades while maintaining trend alignment. Position sizing: 0.25 entry, 0.125 at 1.5R.
Stoploss: 2.5*ATR trailing. Must ensure ≥10 trades per symbol (loose entry filters).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_rsi_pullback_v1"
timeframe = "15m"
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
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    trend = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower_band[0]
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            trend[i] = 1
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            trend[i] = -1
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    return supertrend, trend

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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

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
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
    # EMA for additional trend confirmation
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    
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
    
    for i in range(100, n):
        # 4h trend filter (HTF) - price relative to 4h HMA
        daily_bullish = close[i] > hma_4h_aligned[i]
        daily_bearish = close[i] < hma_4h_aligned[i]
        
        # 15m Supertrend signals
        st_long = st_trend[i] == 1
        st_short = st_trend[i] == -1
        
        # Supertrend flip detection
        st_flip_long = st_trend[i] == 1 and (i > 0 and st_trend[i-1] == -1)
        st_flip_short = st_trend[i] == -1 and (i > 0 and st_trend[i-1] == 1)
        
        # EMA trend confirmation
        ema_trend_long = ema_50[i] > ema_200[i]
        ema_trend_short = ema_50[i] < ema_200[i]
        
        # RSI pullback zone (not extreme - allows more entries)
        rsi_ok_long = rsi[i] < 65  # Not overbought
        rsi_ok_short = rsi[i] > 35  # Not oversold
        
        # RSI pullback entry (RSI dipped then recovering)
        rsi_pullback_long = rsi[i] > 40 and rsi[i] < 60
        rsi_pullback_short = rsi[i] > 40 and rsi[i] < 60
        
        new_signal = 0.0
        
        # LONG ENTRY conditions (multiple paths to ensure trades)
        # Condition 1: Supertrend flip + 4h bullish + RSI OK
        if st_flip_long and daily_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Condition 2: Supertrend long + 4h bullish + EMA trend + RSI pullback
        elif st_long and daily_bullish and ema_trend_long and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Condition 3: Supertrend long + 4h bullish + price > EMA50
        elif st_long and daily_bullish and close[i] > ema_50[i]:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Condition 1: Supertrend flip + 4h bearish + RSI OK
        if st_flip_short and daily_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Condition 2: Supertrend short + 4h bearish + EMA trend + RSI pullback
        elif st_short and daily_bearish and ema_trend_short and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Condition 3: Supertrend short + 4h bearish + price < EMA50
        elif st_short and daily_bearish and close[i] < ema_50[i]:
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
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 1.5 * risk:
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
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
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