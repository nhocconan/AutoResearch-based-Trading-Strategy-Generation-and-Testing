#!/usr/bin/env python3
"""
Experiment #195: 1h HMA Trend + RSI Pullback with 4h/1d Confirmation
Hypothesis: 1h timeframe offers good balance between signal frequency and noise.
Using 4h HMA for intermediate trend + 1d HMA for macro bias. RSI(14) pullback
entries (RSI 40-50 for longs, 50-60 for shorts) avoid extreme mean-reversion
traps that failed in CRSI experiments. Volume filter confirms genuine moves.
ATR trailing stop at 2.5*ATR protects capital. Position size 0.25/0.125 discrete.
Target: Beat Sharpe=0.499 from current best (12h supertrend).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_volume_4h_1d_atr_v1"
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

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    macd_hist = calculate_macd_histogram(close, 12, 26, 9)
    
    # Price vs 1h HMA for local trend
    hma_1h = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track position state (NOT from signals array)
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        trend_1d_bull = close[i] > hma_1d_aligned[i]
        trend_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Local 1h trend
        trend_1h_bull = close[i] > hma_1h[i]
        trend_1h_bear = close[i] < hma_1h[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.2 * vol_sma[i]
        
        # MACD momentum
        macd_bull = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1]
        macd_bear = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1]
        
        # RSI pullback zones (not extremes - learned from CRSI failures)
        rsi_pullback_long = 40 <= rsi[i] <= 55
        rsi_pullback_short = 45 <= rsi[i] <= 60
        
        # RSI momentum confirmation
        rsi_rising = rsi[i] > rsi[i-1]
        rsi_falling = rsi[i] < rsi[i-1]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: 4h bullish + RSI pullback + volume + MACD turning up
        if trend_4h_bull and rsi_pullback_long and rsi_rising:
            if volume_confirmed or macd_bull:
                new_signal = SIZE_ENTRY
        # Secondary: 1d bullish + 1h trend + RSI recovering
        elif trend_1d_bull and trend_1h_bull and rsi[i] > 45 and rsi_rising:
            if macd_bull:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: 4h bearish + RSI pullback + volume + MACD turning down
        if trend_4h_bear and rsi_pullback_short and rsi_falling:
            if volume_confirmed or macd_bear:
                new_signal = -SIZE_ENTRY
        # Secondary: 1d bearish + 1h trend + RSI weakening
        elif trend_1d_bear and trend_1h_bear and rsi[i] < 55 and rsi_falling:
            if macd_bear:
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