#!/usr/bin/env python3
"""
Experiment #287: 12h Donchian Breakout with Daily/Weekly HMA Trend Filter
Hypothesis: Donchian breakouts capture momentum moves better than KAMA/EMA crossovers.
Daily HMA provides trend bias (only trade breakouts in trend direction). Weekly HMA
adds macro confirmation for stronger signals. RSI filter ensures we're not entering
at extreme overbought/oversold levels. Simpler logic than #275 to ensure sufficient trades.
Position sizing: 0.25 entry, 0.125 half at 2R profit. Stoploss: 2.5*ATR trailing.
Target: Beat Sharpe=0.499 from current best (mtf_12h_supertrend_daily_hma_rsi_pullback_v2)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_daily_weekly_hma_rsi_atr_v3"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Track previous values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    
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
        
        # HTF trend filters
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] and prev_close[i] <= donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1] and prev_close[i] >= donchian_lower[i-1]
        
        # RSI filter (not too extreme)
        rsi_ok_long = 30 < rsi[i] < 70
        rsi_ok_short = 30 < rsi[i] < 70
        rsi_pullback_long = 35 < rsi[i] < 55
        rsi_pullback_short = 45 < rsi[i] < 65
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Donchian breakout + Daily HMA bullish + RSI ok
        if breakout_long and daily_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Donchian breakout + Weekly bullish (stronger signal)
        elif breakout_long and weekly_bullish and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Pullback entry in uptrend
        elif daily_bullish and rsi_pullback_long and close[i] > hma_1d_aligned[i]:
            new_signal = SIZE_ENTRY
        # Strong trend (both HTF bullish)
        elif daily_bullish and weekly_bullish and rsi[i] > 45 and close[i] > donchian_upper[i-1] * 0.98:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Donchian breakout + Daily HMA bearish + RSI ok
        if breakout_short and daily_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Donchian breakout + Weekly bearish (stronger signal)
        elif breakout_short and weekly_bearish and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Pullback entry in downtrend
        elif daily_bearish and rsi_pullback_short and close[i] < hma_1d_aligned[i]:
            new_signal = -SIZE_ENTRY
        # Strong trend (both HTF bearish)
        elif daily_bearish and weekly_bearish and rsi[i] < 55 and close[i] < donchian_lower[i-1] * 1.02:
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