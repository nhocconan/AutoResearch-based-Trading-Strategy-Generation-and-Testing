#!/usr/bin/env python3
"""
Experiment #146: 30m Multi-Timeframe Trend-Follow with RSI Pullback
Hypothesis: 30m timeframe offers balance between noise reduction and trade frequency.
Use 4h HMA as primary trend filter (proven in best strategies), 30m RSI for pullback
entries in trend direction. Simpler than regime-switching approaches that failed.
Key insight from failures: too many conflicting filters = whipsaw or 0 trades.
This uses: 4h HMA slope + 30m RSI pullback + ATR stoploss.
Position sizing: 0.30 entry, stoploss at 2.5*ATR, discrete levels to minimize fees.
Timeframe: 30m for adequate trade frequency while avoiding 5m/15m noise.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_4h_hma_rsi_pullback_atr_v1"
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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h_fast = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slow = calculate_hma(df_4h['close'].values, 55)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_fast)
    hma_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slow)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    hma_30m = calculate_hma(close, 21)
    sma_200 = calculate_sma(close, 200)
    
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
    
    for i in range(250, n):  # Start after 200 SMA warmup + buffer
        # 4h HTF trend filter (major trend direction)
        hma_4h_bullish = hma_4h_fast_aligned[i] > hma_4h_slow_aligned[i]
        hma_4h_bearish = hma_4h_fast_aligned[i] < hma_4h_slow_aligned[i]
        
        # 4h HMA slope confirmation (trend momentum)
        hma_4h_slope_up = hma_4h_fast_aligned[i] > hma_4h_fast_aligned[i-3] if i > 3 else False
        hma_4h_slope_down = hma_4h_fast_aligned[i] < hma_4h_fast_aligned[i-3] if i > 3 else False
        
        # 30m LTF trend filter
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # 30m HMA trend
        hma_30m_bullish = close[i] > hma_30m[i]
        hma_30m_bearish = close[i] < hma_30m[i]
        
        # RSI pullback conditions (entry timing)
        rsi_pullback_long = 35 <= rsi[i] <= 50  # Pullback in uptrend
        rsi_pullback_short = 50 <= rsi[i] <= 65  # Pullback in downtrend
        rsi_strong_long = rsi[i] > 55  # Momentum continuation
        rsi_strong_short = rsi[i] < 45  # Momentum continuation
        
        new_signal = 0.0
        
        # LONG entries: 4h bullish + 30m confirmation + RSI timing
        if hma_4h_bullish and hma_4h_slope_up:
            # Pullback entry (preferred - better risk/reward)
            if rsi_pullback_long and hma_30m_bullish and price_above_sma200:
                new_signal = SIZE_ENTRY
            # Momentum entry (when trend is strong)
            elif rsi_strong_long and hma_30m_bullish and hma_4h_slope_up:
                new_signal = SIZE_ENTRY
        
        # SHORT entries: 4h bearish + 30m confirmation + RSI timing
        elif hma_4h_bearish and hma_4h_slope_down:
            # Pullback entry (preferred - better risk/reward)
            if rsi_pullback_short and hma_30m_bearish and price_below_sma200:
                new_signal = -SIZE_ENTRY
            # Momentum entry (when trend is strong)
            elif rsi_strong_short and hma_30m_bearish and hma_4h_slope_down:
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