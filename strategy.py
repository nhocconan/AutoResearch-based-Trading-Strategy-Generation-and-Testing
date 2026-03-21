#!/usr/bin/env python3
"""
Experiment #142: 4h Volatility Breakout with Daily/Weekly Trend Filter
Hypothesis: 4h timeframe captures medium-term moves without excessive noise.
Volatility contraction (BB squeeze) followed by breakout produces strong directional moves.
Daily HMA provides primary trend filter, Weekly HMA provides macro confirmation.
Only trade breakouts in direction of HTF trend to avoid whipsaws.
Volume confirmation ensures breakout has participation. Time-based exit prevents
giving back profits in choppy conditions. This differs from failed 4h strategies
by focusing on volatility breakouts rather than regime detection or simple trend following.
Position sizing: 0.25 entry, reduce to 0.12 at 1.5R, stoploss at 2.5*ATR trailing.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_breakout_daily_weekly_hma_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth percentile."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    # Calculate bandwidth percentile over 100 bars
    bw_percentile = pd.Series(bandwidth).rolling(window=100, min_periods=50).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0.5,
        raw=False
    ).values
    return upper, lower, bandwidth, bw_percentile

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

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

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
    bb_upper, bb_lower, bb_bw, bb_bw_pct = calculate_bollinger_bands(close, 20, 2.0)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Calculate price momentum (ROC 10)
    roc = np.zeros(n)
    for i in range(10, n):
        roc[i] = (close[i] - close[i-10]) / close[i-10] * 100 if close[i-10] > 0 else 0
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss and time exit
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Strong trend: both daily and weekly agree
        strong_bullish = daily_bullish and weekly_bullish
        strong_bearish = daily_bearish and weekly_bearish
        
        # Volatility squeeze detection (BB bandwidth in lower 20th percentile)
        squeeze = bb_bw_pct[i] < 0.25 if not np.isnan(bb_bw_pct[i]) else False
        
        # Volume confirmation (current volume > 1.5x 20-bar average)
        volume_confirmed = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        # Price position relative to BB
        bb_range = bb_upper[i] - bb_lower[i]
        bb_position = (close[i] - bb_lower[i]) / bb_range if bb_range > 0 else 0.5
        
        # Breakout detection
        breakout_long = close[i] > bb_upper[i] and close[i-1] <= bb_upper[i-1]
        breakout_short = close[i] < bb_lower[i] and close[i-1] >= bb_lower[i-1]
        
        # Momentum confirmation
        momentum_long = roc[i] > 2.0
        momentum_short = roc[i] < -2.0
        
        # RSI filter (avoid extreme overbought/oversold on entry)
        rsi_ok_long = rsi[i] < 75
        rsi_ok_short = rsi[i] > 25
        
        new_signal = 0.0
        
        # LONG ENTRY: Squeeze + breakout + volume + trend + momentum
        if breakout_long and volume_confirmed and strong_bullish and rsi_ok_long and momentum_long:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: Squeeze + breakout + volume + trend + momentum
        elif breakout_short and volume_confirmed and strong_bearish and rsi_ok_short and momentum_short:
            new_signal = -SIZE_ENTRY
        
        # Alternative: Trend continuation without squeeze (weaker signal)
        elif strong_bullish and not daily_bearish and rsi[i] > 50 and rsi[i] < 70:
            # Enter on pullback to middle BB
            if bb_position < 0.5 and close[i] > close[i-1]:
                new_signal = SIZE_ENTRY * 0.7  # Smaller position for continuation
        
        elif strong_bearish and not daily_bullish and rsi[i] < 50 and rsi[i] > 30:
            # Enter on pullback to middle BB
            if bb_position > 0.5 and close[i] < close[i-1]:
                new_signal = -SIZE_ENTRY * 0.7
        
        # Stoploss and profit taking logic (Rule 6) - check BEFORE updating position tracking
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
                # Take profit at 1.5R (reduce position)
                risk = 2.5 * atr[entry_bar] if entry_bar > 0 else 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 1.5 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
            
            # Time-based exit: reduce after 20 bars if not profitable
            bars_held = i - entry_bar
            if bars_held > 20 and not position_reduced and profit < 0.5 * risk:
                new_signal = SIZE_HALF
            if bars_held > 40:
                new_signal = 0.0
        
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
                # Take profit at 1.5R (reduce position)
                risk = 2.5 * atr[entry_bar] if entry_bar > 0 else 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 1.5 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
            
            # Time-based exit: reduce after 20 bars if not profitable
            bars_held = i - entry_bar
            if bars_held > 20 and not position_reduced and profit < 0.5 * risk:
                new_signal = -SIZE_HALF
            if bars_held > 40:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            entry_bar = i
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            entry_bar = i
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
            entry_bar = 0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals