#!/usr/bin/env python3
"""
Experiment #159: 1h Dual Momentum with 12h HMA Trend Filter + RSI Pullback
Hypothesis: 1h timeframe offers good balance between signal frequency and noise.
Using 12h HMA (slower than daily) provides more stable trend filter that works
in both bull and bear markets. RSI pullback entries (RSI<40 in uptrend, RSI>60
in downtrend) ensure we're not chasing extremes. MACD histogram confirms momentum
direction. This combines trend-following (works 2021 bull) with pullback entries
(works 2022-2025 bear/range). ATR stoploss at 2.5*ATR protects capital. Position
sizing: 0.25 entry, 0.125 at 2R profit. Discrete levels minimize fee churn.
Key: loosened RSI thresholds (40/60 not 30/70) to ensure sufficient trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_dual_momentum_12h_hma_rsi_macd_atr_v1"
timeframe = "1h"
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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    sma_50 = calculate_sma(close, 50)
    hma_20 = calculate_hma(close, 20)
    hma_50_local = calculate_hma(close, 50)
    
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
        # 12h trend filter (major trend direction)
        hma_12h_valid = hma_12h_aligned[i] > 0
        trend_12h_bullish = hma_12h_valid and close[i] > hma_12h_aligned[i]
        trend_12h_bearish = hma_12h_valid and close[i] < hma_12h_aligned[i]
        
        # 1h trend filter
        trend_1h_bullish = hma_20[i] > hma_50_local[i] and sma_50[i] > 0
        trend_1h_bearish = hma_20[i] < hma_50_local[i] and sma_50[i] > 0
        
        # MACD momentum
        macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1] if i > 0 else False
        macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1] if i > 0 else False
        macd_cross_up = macd_hist[i] > 0 and macd_hist[i-1] <= 0 if i > 0 else False
        macd_cross_down = macd_hist[i] < 0 and macd_hist[i-1] >= 0 if i > 0 else False
        
        # RSI pullback levels (loosened for more trades)
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        rsi_neutral_low = rsi[i] < 50
        rsi_neutral_high = rsi[i] > 50
        rsi_rising = rsi[i] > rsi[i-2] if i > 2 else False
        rsi_falling = rsi[i] < rsi[i-2] if i > 2 else False
        
        new_signal = 0.0
        
        # LONG ENTRY: 12h bullish + RSI pullback + MACD confirmation
        if trend_12h_bullish:
            # Strong long: 12h bullish + RSI pullback + MACD turning up
            if rsi_oversold and (macd_cross_up or macd_bullish):
                new_signal = SIZE_ENTRY
            # Moderate long: 12h bullish + 1h bullish + RSI rising
            elif trend_1h_bullish and rsi_rising and rsi_neutral_low:
                new_signal = SIZE_ENTRY
            # Breakout long: 12h bullish + MACD cross up
            elif macd_cross_up and trend_1h_bullish:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: 12h bearish + RSI pullback + MACD confirmation
        elif trend_12h_bearish:
            # Strong short: 12h bearish + RSI pullback + MACD turning down
            if rsi_overbought and (macd_cross_down or macd_bearish):
                new_signal = -SIZE_ENTRY
            # Moderate short: 12h bearish + 1h bearish + RSI falling
            elif trend_1h_bearish and rsi_falling and rsi_neutral_high:
                new_signal = -SIZE_ENTRY
            # Breakdown short: 12h bearish + MACD cross down
            elif macd_cross_down and trend_1h_bearish:
                new_signal = -SIZE_ENTRY
        
        # NEUTRAL 12h: Use 1h signals only (range market)
        else:
            # Long in range: RSI oversold + MACD cross up
            if rsi_oversold and macd_cross_up:
                new_signal = SIZE_ENTRY
            # Short in range: RSI overbought + MACD cross down
            elif rsi_overbought and macd_cross_down:
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