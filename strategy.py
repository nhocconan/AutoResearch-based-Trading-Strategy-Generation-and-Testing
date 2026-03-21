#!/usr/bin/env python3
"""
Experiment #094: 4h Supertrend with Daily HMA + RSI Pullback (Simplified v2)
Hypothesis: #088 had Sharpe=0.032 with Supertrend+Daily HMA+RSI+ADX on 4h.
The ADX>20 filter was too restrictive. This version: (1) lowers ADX to >15,
(2) loosens RSI to <80/>20 for more entries, (3) adds MACD confirmation for
better entry timing, (4) simpler position tracking to avoid bugs.
Keep proven components: Supertrend for trend, Daily HMA for HTF bias.
Position sizing: 0.30 entry, 0.15 at 1.5R profit. ATR stop at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_daily_hma_rsi_macd_v2"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            if direction[i-1] == 1:
                supertrend[i] = max(supertrend[i], lower_band[i])
                direction[i] = 1
            else:
                supertrend[i] = min(supertrend[i], upper_band[i])
                direction[i] = -1
    
    return supertrend, direction

def calculate_macd_histogram(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram.values

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
    adx = calculate_adx(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    macd_hist = calculate_macd_histogram(close, 12, 26, 9)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Position tracking
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Daily trend filter (HTF)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # Supertrend flip detection
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # MACD confirmation
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        
        # ADX trend strength (lowered from 20 to 15)
        trend_strong = adx[i] > 15
        
        # RSI filter (loosened from 70/30 to 80/20)
        rsi_ok_long = rsi[i] < 80
        rsi_ok_short = rsi[i] > 20
        
        new_signal = 0.0
        
        # LONG ENTRY: Supertrend flip + Daily bullish + MACD + RSI + ADX
        if st_flip_long and daily_bullish and macd_bullish and rsi_ok_long and trend_strong:
            new_signal = SIZE_ENTRY
        # Alternative: Supertrend bullish + Daily bullish + MACD cross up
        elif st_bullish and daily_bullish and macd_hist[i] > macd_hist[i-1] and macd_hist[i] > 0 and rsi_ok_long:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: Supertrend flip + Daily bearish + MACD + RSI + ADX
        if st_flip_short and daily_bearish and macd_bearish and rsi_ok_short and trend_strong:
            new_signal = -SIZE_ENTRY
        # Alternative: Supertrend bearish + Daily bearish + MACD cross down
        elif st_bearish and daily_bearish and macd_hist[i] < macd_hist[i-1] and macd_hist[i] < 0 and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic - check BEFORE position tracking update
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_price:
                highest_price = close[i]
            
            current_stop = highest_price - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                profit = close[i] - entry_price
                risk = 2.5 * atr[i]
                if profit >= 1.5 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_price or lowest_price == 0.0:
                lowest_price = close[i]
            
            current_stop = lowest_price + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                profit = entry_price - close[i]
                risk = 2.5 * atr[i]
                if profit >= 1.5 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_price = close[i] if position_side > 0 else 0.0
            lowest_price = close[i] if position_side < 0 else 0.0
            position_reduced = False
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_price = close[i] if position_side > 0 else 0.0
            lowest_price = close[i] if position_side < 0 else 0.0
            position_reduced = False
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_price = 0.0
            lowest_price = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals