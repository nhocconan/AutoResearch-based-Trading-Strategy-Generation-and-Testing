#!/usr/bin/env python3
"""
Experiment #112: 4h Fisher Transform + Daily HMA Trend + BB Width Regime
Hypothesis: Fisher Transform excels at catching reversals in bear/range markets
(2025 test period). Combined with Daily HMA for trend bias and BB Width to
only trade during volatility expansion (avoid choppy consolidation). This
differs from failed RSI pullback strategies by using Fisher's non-linear
transform which normalizes price better than RSI. Position sizing: 0.30 entry,
stoploss at 2*ATR, take profit at 2R then trail.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_daily_hma_bbwidth_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    hl2 = (high + low) / 2
    hl2_s = pd.Series(hl2)
    
    # Calculate highest high and lowest low over period
    hh = hl2_s.rolling(window=period, min_periods=period).max()
    ll = hl2_s.rolling(window=period, min_periods=period).min()
    
    # Normalize to -1 to +1 range
    norm = 0.66 * ((hl2 - ll) / (hh - ll + 1e-10) - 0.5)
    norm = np.clip(norm, -0.99, 0.99)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + norm) / (1 - norm + 1e-10))
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    
    return fisher, fisher_prev

def calculate_bb_width(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width for volatility regime detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / sma
    
    # Calculate percentile rank over 100 bars
    bb_width_s = pd.Series(bb_width)
    bb_width_pct = bb_width_s.rolling(window=100, min_periods=50).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False
    )
    
    return bb_width.values, bb_width_pct.values

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
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    bb_width, bb_width_pct = calculate_bb_width(close, 20, 2.0)
    
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
        # Daily trend filter (HTF) - price relative to Daily HMA
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Fisher Transform signals
        fisher_cross_long = fisher[i] > -1.5 and fisher_prev[i] <= -1.5
        fisher_cross_short = fisher[i] < 1.5 and fisher_prev[i] >= 1.5
        
        # Fisher extreme levels (mean reversion)
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        # BB Width regime - only trade when volatility expanding
        vol_expanding = bb_width_pct[i] > 0.5  # Above median volatility
        
        # RSI filter (avoid extremes for trend entries)
        rsi_ok_long = rsi[i] < 70
        rsi_ok_short = rsi[i] > 30
        
        new_signal = 0.0
        
        # LONG ENTRY conditions
        # Condition 1: Fisher cross + Daily bullish + Vol expanding
        if fisher_cross_long and daily_bullish and vol_expanding:
            new_signal = SIZE_ENTRY
        # Condition 2: Fisher oversold + Daily bullish (mean reversion in uptrend)
        elif fisher_oversold and daily_bullish and rsi[i] < 50:
            new_signal = SIZE_ENTRY
        # Condition 3: Fisher cross + RSI ok + Vol expanding (trend agnostic)
        elif fisher_cross_long and rsi_ok_long and vol_expanding and not daily_bearish:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Condition 1: Fisher cross + Daily bearish + Vol expanding
        if fisher_cross_short and daily_bearish and vol_expanding:
            new_signal = -SIZE_ENTRY
        # Condition 2: Fisher overbought + Daily bearish (mean reversion in downtrend)
        elif fisher_overbought and daily_bearish and rsi[i] > 50:
            new_signal = -SIZE_ENTRY
        # Condition 3: Fisher cross + RSI ok + Vol expanding (trend agnostic)
        elif fisher_cross_short and rsi_ok_short and vol_expanding and not daily_bullish:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                profit = close[i] - entry_price
                risk = 2.0 * atr[i]
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                profit = entry_price - close[i]
                risk = 2.0 * atr[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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