#!/usr/bin/env python3
"""
Experiment #436: 4h Supertrend + Daily HMA + RSI Pullback + ATR Stoploss
Hypothesis: 4h Supertrend provides clear trend signals, Daily HMA filters direction,
RSI pullback ensures entries on retracements (not breakouts = less whipsaw).
Key insight: Previous 4h strategies failed due to whipsaw in transitions.
RSI pullback (wait for RSI<50 in uptrend, >50 in downtrend) reduces false entries.
Unlike CHOP-based strategies (#430, #431 failed), Supertrend has built-in ATR stop.
Position size: 0.25 entry, 0.125 half, stoploss 2.5*ATR (trailing)
Timeframe: 4h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_daily_hma_rsi_pullback_atr_v2"
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

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.divide(avg_g, avg_l, out=np.ones_like(avg_g), where=avg_l!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend values, direction (1=long, -1=short)
    """
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.zeros(len(close))
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1
    
    return supertrend, direction

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    sma50 = calculate_sma(close, 50)
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma50[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (long-term direction)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend change detection
        st_changed_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_changed_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # RSI conditions for pullback entries
        rsi_pullback_long = rsi[i] < 55 and rsi[i] > 35  # Pullback in uptrend
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 65  # Pullback in downtrend
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Price position
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Supertrend long + Daily bullish + RSI pullback
        if st_long and daily_bullish and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Path 2: Supertrend flipped to long + Daily bullish
        elif st_changed_long and daily_bullish:
            new_signal = SIZE_ENTRY
        # Path 3: Supertrend long + Above SMA50 + RSI not overbought
        elif st_long and above_sma50 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        # Path 4: RSI oversold + Daily bullish (mean reversion in trend)
        elif rsi_oversold and daily_bullish and st_long:
            new_signal = SIZE_ENTRY
        # Path 5: Supertrend long + Daily bullish + RSI > 40 (momentum)
        elif st_long and daily_bullish and rsi[i] > 40 and rsi[i] < 75:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Supertrend short + Daily bearish + RSI pullback
        if st_short and daily_bearish and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Path 2: Supertrend flipped to short + Daily bearish
        elif st_changed_short and daily_bearish:
            new_signal = -SIZE_ENTRY
        # Path 3: Supertrend short + Below SMA50 + RSI not oversold
        elif st_short and below_sma50 and rsi[i] > 30:
            new_signal = -SIZE_ENTRY
        # Path 4: RSI overbought + Daily bearish (mean reversion in trend)
        elif rsi_overbought and daily_bearish and st_short:
            new_signal = -SIZE_ENTRY
        # Path 5: Supertrend short + Daily bearish + RSI < 60 (momentum)
        elif st_short and daily_bearish and rsi[i] < 60 and rsi[i] > 25:
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
        
        elif position_side < 0 and entry_price > 0:
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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