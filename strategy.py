#!/usr/bin/env python3
"""
Experiment #131: 12h Supertrend + Daily HMA + RSI Pullback + ADX Filter
Hypothesis: Current best (mtf_12h_supertrend_daily_hma_rsi_pullback_v2) uses Supertrend successfully.
Problem with #125 (KAMA) was too many conflicting entry paths causing low trade frequency.
Solution: SIMPLER logic - Supertrend for trend direction, Daily HMA as filter (not Weekly - too restrictive),
RSI pullback (30-70 range) for entries, ADX > 20 to avoid choppy markets.
Fewer conditions = more trades while maintaining quality. ADX filter critical for 12h timeframe.
Position sizing: 0.30 entry, 0.15 at 2R profit, 2.5*ATR trailing stop.
Timeframe: 12h (mandatory for this experiment) with 1d HTF reference.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_supertrend_daily_hma_rsi_adx_v1"
timeframe = "12h"
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
    Returns: supertrend values, direction (1=long, -1=short)
    """
    atr = calculate_atr(high, low, close, period)
    n = len(close)
    
    hl2 = (high + low) / 2.0
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish (price above ST), -1 = bearish
    
    supertrend[0] = lower_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            # Check if trend changed
            if direction[i-1] == 1 and close[i] < lower_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            elif direction[i-1] == -1 and close[i] > upper_band[i]:
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = pd.Series(100 * plus_dm / np.where(atr > 0, atr, 1)).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(100 * minus_dm / np.where(atr > 0, atr, 1)).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filter (Daily HMA)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend change detection (entry trigger)
        st_change_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_change_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # RSI pullback zones (wider range for more trades)
        rsi_ok_long = 30 <= rsi[i] <= 70
        rsi_ok_short = 30 <= rsi[i] <= 70
        
        # RSI momentum
        rsi_momentum_long = rsi[i] > 40
        rsi_momentum_short = rsi[i] < 60
        
        # ADX trend strength filter (avoid choppy markets)
        trend_strong = adx[i] > 20
        
        new_signal = 0.0
        
        # LONG ENTRY: Simpler conditions for more trades
        # Path 1: Supertrend flip long + Daily bullish + ADX strong + RSI ok
        if st_change_long and daily_bullish and trend_strong and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Path 2: Already long on Supertrend + Daily bullish + RSI pullback (add-on)
        elif st_long and daily_bullish and rsi[i] < 50 and rsi_momentum_long and signals[i-1] == 0:
            new_signal = SIZE_ENTRY
        # Path 3: Strong trend continuation (ADX very strong)
        elif st_long and daily_bullish and adx[i] > 30 and rsi[i] > 45 and signals[i-1] == 0:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY
        # Path 1: Supertrend flip short + Daily bearish + ADX strong + RSI ok
        if st_change_short and daily_bearish and trend_strong and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Path 2: Already short on Supertrend + Daily bearish + RSI pullback (add-on)
        elif st_short and daily_bearish and rsi[i] > 50 and rsi_momentum_short and signals[i-1] == 0:
            new_signal = -SIZE_ENTRY
        # Path 3: Strong trend continuation (ADX very strong)
        elif st_short and daily_bearish and adx[i] > 30 and rsi[i] < 55 and signals[i-1] == 0:
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