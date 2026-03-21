#!/usr/bin/env python3
"""
Experiment #434: 30m Supertrend + 4h HMA Trend + RSI Filter
Hypothesis: 30m Supertrend captures intraday momentum while 4h HMA prevents
counter-trend trades. Simplified entry conditions ensure sufficient trade count.
Key insight: Previous 30m strategies failed due to over-filtering. This uses
fewer conditions to generate more trades while keeping 4h trend as primary filter.
Timeframe: 30m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Position size: 0.25 discrete, stoploss 2*ATR emergency + Supertrend exit.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_4h_hma_rsi_simple_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    trend = np.ones(n)
    
    supertrend[0] = upper_band[0]
    
    for i in range(1, n):
        if trend[i-1] == 1:
            if close[i] < lower_band[i]:
                trend[i] = -1
                supertrend[i] = upper_band[i]
            else:
                trend[i] = 1
                supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            if close[i] > upper_band[i]:
                trend[i] = 1
                supertrend[i] = lower_band[i]
            else:
                trend[i] = -1
                supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    return supertrend, trend

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align 4h HMA to 30m (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, atr, 3.0)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(st_trend[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (primary filter)
        bullish_4h = close[i] > hma_4h_aligned[i]
        bearish_4h = close[i] < hma_4h_aligned[i]
        
        # Supertrend signals
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # Supertrend flips (entry triggers)
        st_flip_long = st_bullish and st_trend[i-1] == -1
        st_flip_short = st_bearish and st_trend[i-1] == 1
        
        new_signal = 0.0
        
        # === LONG ENTRIES (simplified for more trades) ===
        # Path 1: Supertrend flip long + 4h bullish (primary entry)
        if st_flip_long and bullish_4h:
            new_signal = SIZE
        # Path 2: Already bullish Supertrend + 4h bullish + RSI momentum
        elif st_bullish and bullish_4h and rsi[i] > 50 and rsi[i] < 75:
            new_signal = SIZE
        # Path 3: RSI pullback in uptrend (RSI was <45, now >45)
        elif st_bullish and bullish_4h and rsi[i] > 45 and rsi[i-1] < 45:
            new_signal = SIZE
        # Path 4: Simple trend - Supertrend bullish + 4h bullish
        elif st_bullish and bullish_4h:
            new_signal = SIZE
        
        # === SHORT ENTRIES (simplified for more trades) ===
        # Path 1: Supertrend flip short + 4h bearish (primary entry)
        if st_flip_short and bearish_4h:
            new_signal = -SIZE
        # Path 2: Already bearish Supertrend + 4h bearish + RSI momentum
        elif st_bearish and bearish_4h and rsi[i] < 50 and rsi[i] > 25:
            new_signal = -SIZE
        # Path 3: RSI rejection in downtrend (RSI was >55, now <55)
        elif st_bearish and bearish_4h and rsi[i] < 55 and rsi[i-1] > 55:
            new_signal = -SIZE
        # Path 4: Simple trend - Supertrend bearish + 4h bearish
        elif st_bearish and bearish_4h:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals