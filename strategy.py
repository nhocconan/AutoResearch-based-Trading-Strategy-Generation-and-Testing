#!/usr/bin/env python3
"""
Experiment #136: 4h Donchian Breakout with Dual HTF Trend Filter + Volume Confirmation
Hypothesis: 4h timeframe captures medium-term trends better than 1h but with more
signals than 1d. Donchian breakouts (20-period) are proven on longer timeframes.
Key innovations: (1) Dual HTF filter - 1d HMA for intermediate trend, 1w HMA for
major trend direction. (2) Volume confirmation - breakout must have 1.5x avg volume.
(3) RSI pullback entry - wait for pullback after breakout confirmation, not chase.
(4) Conservative sizing 0.25 with 2.5*ATR stoploss. This should work in both bull
(2021) and bear (2022, 2025) markets by following the major trend direction.
Timeframe: 4h | HTF: 1d, 1w | Expected: 30-50 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_volume_1d_1w_hma_rsi_pullback_v1"
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

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume for volume confirmation."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian_channels(high, low, 20)
    vol_sma = calculate_volume_sma(volume, 20)
    
    # HMA for trend confirmation on 4h
    hma_4h_fast = calculate_hma(close, 10)
    hma_4h_slow = calculate_hma(close, 30)
    
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
    
    # Track breakout confirmation
    breakout_confirmed_long = False
    breakout_confirmed_short = False
    breakout_bar_long = -100
    breakout_bar_short = -100
    
    for i in range(100, n):
        # HTF trend filters
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # 4h HMA trend
        hma_trend_long = hma_4h_fast[i] > hma_4h_slow[i]
        hma_trend_short = hma_4h_fast[i] < hma_4h_slow[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_sma[i] if vol_sma[i] > 0 else False
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # RSI pullback conditions (wait for pullback after breakout)
        rsi_pullback_long = 40 <= rsi[i] <= 55  # Pullback in uptrend
        rsi_pullback_short = 45 <= rsi[i] <= 60  # Pullback in downtrend
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 50
        rsi_momentum_short = rsi[i] < 50
        
        new_signal = 0.0
        
        # LONG ENTRY: Donchian breakout + Volume + HTF trend + RSI confirmation
        if breakout_long and volume_confirmed:
            breakout_confirmed_long = True
            breakout_bar_long = i
        
        if breakout_confirmed_long and (i - breakout_bar_long) <= 10:
            # Wait for pullback entry after breakout confirmation
            if rsi_pullback_long and daily_bullish and weekly_bullish and hma_trend_long:
                new_signal = SIZE_ENTRY
                breakout_confirmed_long = False  # Reset after entry
        
        # Also enter on strong momentum breakout if all filters align
        if breakout_long and volume_confirmed and rsi_momentum_long and daily_bullish and weekly_bullish:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: Donchian breakdown + Volume + HTF trend + RSI confirmation
        if breakout_short and volume_confirmed:
            breakout_confirmed_short = True
            breakout_bar_short = i
        
        if breakout_confirmed_short and (i - breakout_bar_short) <= 10:
            # Wait for pullback entry after breakdown confirmation
            if rsi_pullback_short and daily_bearish and weekly_bearish and hma_trend_short:
                new_signal = -SIZE_ENTRY
                breakout_confirmed_short = False  # Reset after entry
        
        # Also enter on strong momentum breakdown if all filters align
        if breakout_short and volume_confirmed and rsi_momentum_short and daily_bearish and weekly_bearish:
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