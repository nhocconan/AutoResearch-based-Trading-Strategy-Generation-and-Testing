#!/usr/bin/env python3
"""
Experiment #249: 1h RSI Mean Reversion with 4h/1d HMA Trend Filter
Hypothesis: 1h timeframe with STRONG higher-timeframe filters can work if we:
(1) Only trade when 4h AND 1d trend align (reduces whipsaw),
(2) Use RSI extremes for counter-trend entries within the trend (buy dips in uptrend),
(3) Fewer but higher-conviction trades (RSI 40/60 thresholds, not 30/70),
(4) Wide 3*ATR stops to avoid noise,
(5) Discrete position sizing (0.25 entry, 0.125 half) to minimize fee churn.

This differs from failed 1h strategies by using STRONGER HTF alignment (both 4h AND 1d must agree)
and simpler entry logic (RSI threshold only, no conflicting MACD/ADX filters).
Target: Beat Sharpe=0.499 from current best (mtf_12h_supertrend_daily_hma_rsi_pullback_v2).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_extreme_4h_1d_hma_aligned_atr_v1"
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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma_50 = calculate_sma(close, 50)
    
    # Previous RSI for cross detection
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Position tracking
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # === HTF TREND FILTERS (both must align) ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        trend_1d_bull = close[i] > hma_1d_aligned[i]
        trend_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong trend alignment (both 4h and 1d agree)
        strong_bull = trend_4h_bull and trend_1d_bull
        strong_bear = trend_4h_bear and trend_1d_bear
        
        # === RSI MOMENTUM SIGNALS (looser thresholds for more trades) ===
        # Long: RSI was oversold (<40) and now rising (>40)
        rsi_long_setup = prev_rsi[i] < 40 and rsi[i] >= 40
        # Short: RSI was overbought (>60) and now falling (<60)
        rsi_short_setup = prev_rsi[i] > 60 and rsi[i] <= 60
        
        # === 1H SMA CONFIRMATION ===
        above_sma = close[i] > sma_50[i]
        below_sma = close[i] < sma_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Strong uptrend + RSI recovery from oversold
        if strong_bull and rsi_long_setup:
            new_signal = SIZE_ENTRY
        # Moderate uptrend (4h only) + RSI recovery + above SMA
        elif trend_4h_bull and rsi_long_setup and above_sma:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Strong downtrend + RSI decline from overbought
        if strong_bear and rsi_short_setup:
            new_signal = -SIZE_ENTRY
        # Moderate downtrend (4h only) + RSI decline + below SMA
        elif trend_4h_bear and rsi_short_setup and below_sma:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS & TAKE PROFIT LOGIC ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing stop
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Trail stop: 3*ATR from highest
            current_stop = highest_close - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            # Take profit at 2R (reduce to half)
            elif not position_reduced:
                risk = 3.0 * atr[int(np.clip(i - 50, 0, n-1))]  # approx entry ATR
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing stop
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            # Trail stop: 3*ATR from lowest
            current_stop = lowest_close + 3.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            # Take profit at 2R (reduce to half)
            elif not position_reduced:
                risk = 3.0 * atr[int(np.clip(i - 50, 0, n-1))]  # approx entry ATR
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # === UPDATE POSITION TRACKING ===
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
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