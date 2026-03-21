#!/usr/bin/env python3
"""
Experiment #216: 1d Donchian Breakout with Weekly HMA Trend Filter
Hypothesis: Daily Donchian breakouts (20-period) capture multi-week momentum moves.
Weekly HMA provides macro trend bias (only long when price > 1w HMA, only short when <).
RSI(14) filter avoids entering at extremes (>70 or <30). ATR(14) trailing stop at 2.5*ATR.
Position sizing: 0.30 entry, reduce to 0.15 at 2R profit. This is simpler than regime-switching
and should generate consistent trades on daily timeframe. Target: Beat Sharpe=0.499.
Why 1d: Daily bars filter noise, capture sustained moves, fewer trades = less fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_hma_rsi_atr_v1"
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
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    # Track previous Donchian values for breakout detection
    prev_donchian_upper = np.roll(donchian_upper, 1)
    prev_donchian_lower = np.roll(donchian_lower, 1)
    prev_donchian_upper[0] = donchian_upper[0]
    prev_donchian_lower[0] = donchian_lower[0]
    
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
        # Weekly trend filter
        weekly_bullish = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else False
        weekly_bearish = close[i] < hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else False
        
        # RSI filter (avoid extremes)
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        rsi_bullish = rsi[i] > 40
        rsi_bearish = rsi[i] < 60
        
        # Donchian breakout detection
        breakout_long = close[i] > prev_donchian_upper[i] and close[i-1] <= prev_donchian_upper[i-1]
        breakout_short = close[i] < prev_donchian_lower[i] and close[i-1] >= prev_donchian_lower[i-1]
        
        # Donchian continuation (price above/below mid)
        above_mid = close[i] > donchian_mid[i]
        below_mid = close[i] < donchian_mid[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Breakout long with weekly trend confirmation
        if breakout_long and weekly_bullish and rsi_not_overbought:
            new_signal = SIZE_ENTRY
        
        # Continuation long (price above Donchian mid + weekly bullish)
        elif above_mid and weekly_bullish and rsi_bullish and rsi_not_overbought:
            # Enter on pullback to mid
            if i > 1 and close[i-1] < donchian_mid[i-1] and close[i] > donchian_mid[i]:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Breakout short with weekly trend confirmation
        if breakout_short and weekly_bearish and rsi_not_oversold:
            new_signal = -SIZE_ENTRY
        
        # Continuation short (price below Donchian mid + weekly bearish)
        elif below_mid and weekly_bearish and rsi_bearish and rsi_not_oversold:
            # Enter on pullback to mid
            if i > 1 and close[i-1] > donchian_mid[i-1] and close[i] < donchian_mid[i]:
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