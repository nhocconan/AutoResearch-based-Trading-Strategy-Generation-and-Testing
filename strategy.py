#!/usr/bin/env python3
"""
Experiment #143: 12h Donchian Breakout with Daily/Weekly HMA Trend Filter
Hypothesis: 12h timeframe captures medium-term trends while avoiding noise.
Donchian Channel (20-period) breakout provides clear trend entry signals.
Daily HMA filters major trend direction, Weekly HMA confirms long-term bias.
RSI pullback (40-60 range) ensures we enter on retracement, not exhaustion.
This should work in both bull (2021) and bear (2022, 2025) markets by
following the HTF trend direction. Position sizing: 0.25 entry, 0.125 at 2R.
Stoploss: 2.5*ATR trailing stop. Target: Sharpe > 0.5, trades > 30/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_daily_weekly_hma_rsi_pullback_v1"
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

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smooth with Wilder's method
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI calculations
    plus_di = np.where(tr_s > 0, 100 * plus_dm_s / tr_s, 0)
    minus_di = np.where(tr_s > 0, 100 * minus_dm_s / tr_s, 0)
    
    # DX and ADX
    di_sum = plus_di + minus_di
    dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    adx = calculate_adx(high, low, close, 14)
    donch_upper, donch_lower, donch_middle = calculate_donchian_channels(high, low, 20)
    
    # Additional trend filter - EMA crossover
    ema_fast = pd.Series(close).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, min_periods=26, adjust=False).mean().values
    
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
        # Weekly trend filter (major trend direction)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend filter
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # EMA crossover trend
        ema_trend_long = ema_fast[i] > ema_slow[i]
        ema_trend_short = ema_fast[i] < ema_slow[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donch_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donch_lower[i-1] if i > 0 else False
        
        # Pullback entry (price near Donchian middle after breakout)
        pullback_long = close[i] < donch_middle[i] and close[i] > donch_lower[i]
        pullback_short = close[i] > donch_middle[i] and close[i] < donch_upper[i]
        
        # RSI conditions for entry timing
        rsi_bullish = rsi[i] > 45 and rsi[i] < 70
        rsi_bearish = rsi[i] < 55 and rsi[i] > 30
        
        # ADX trend strength filter
        trend_strong = adx[i] > 20
        trend_weak = adx[i] < 25
        
        new_signal = 0.0
        
        # LONG ENTRY: Donchian breakout + HTF bullish + RSI confirmation
        if weekly_bullish and daily_bullish:
            # Breakout entry with strong trend
            if breakout_long and trend_strong and rsi_bullish:
                new_signal = SIZE_ENTRY
            # Pullback entry in established uptrend
            elif ema_trend_long and pullback_long and rsi[i] > 40 and rsi[i] < 60:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: Donchian breakdown + HTF bearish + RSI confirmation
        elif weekly_bearish and daily_bearish:
            # Breakdown entry with strong trend
            if breakout_short and trend_strong and rsi_bearish:
                new_signal = -SIZE_ENTRY
            # Pullback entry in established downtrend
            elif ema_trend_short and pullback_short and rsi[i] > 40 and rsi[i] < 60:
                new_signal = -SIZE_ENTRY
        
        # NEUTRAL/TRANSITION: EMA crossover with ADX confirmation
        else:
            # Long on EMA cross up with momentum
            if ema_trend_long and ema_fast[i-1] <= ema_slow[i-1] and adx[i] > 18 and rsi[i] > 50:
                new_signal = SIZE_ENTRY
            # Short on EMA cross down with momentum
            elif ema_trend_short and ema_fast[i-1] >= ema_slow[i-1] and adx[i] > 18 and rsi[i] < 50:
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