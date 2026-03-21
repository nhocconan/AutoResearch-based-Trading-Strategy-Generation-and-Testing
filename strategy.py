#!/usr/bin/env python3
"""
Experiment #005: 12h HMA Trend + Daily Filter + RSI Pullback
Hypothesis: 12h timeframe captures multi-day swings with less noise than intraday.
Daily HMA provides major trend regime filter (bull/bear).
12h HMA crossover gives entry signals with RSI pullback confirmation.
ATR-based stoploss (2.5x) protects against crashes like 2022.
Position sizing capped at 0.30 with discrete levels to minimize fee churn.
Relaxed entry conditions to ensure ≥10 trades/symbol on train data.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_daily_v1"
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
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper[0]
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        else:
            supertrend[i] = upper[i]
            direction[i] = -1
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    hma_fast = calculate_hma(close, 16)
    hma_slow = calculate_hma(close, 48)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=1.0)
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trailing_stop = np.zeros(n)
    
    for i in range(100, n):
        # Daily trend filter (major regime)
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # 12h HMA trend
        hma_trend_long = hma_fast[i] > hma_slow[i]
        hma_trend_short = hma_fast[i] < hma_slow[i]
        
        # HMA crossover signals
        hma_cross_long = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_cross_short = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        
        # Supertrend confirmation
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # RSI pullback entries (relaxed for more trades)
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 65  # Not extreme
        rsi_pullback_short = rsi[i] > 35 and rsi[i] < 65  # Same range
        rsi_rising = rsi[i] > rsi[i-5] if i > 5 else True
        rsi_falling = rsi[i] < rsi[i-5] if i > 5 else True
        
        # Volume confirmation (relaxed)
        vol_confirm = volume[i] > vol_sma[i] * 0.7 if vol_sma[i] > 0 else True
        
        # Entry logic - relaxed conditions to ensure trades
        new_signal = 0.0
        
        # Long entry: daily bullish + HMA trend + Supertrend + RSI ok
        if daily_bullish and hma_trend_long and st_long and rsi_pullback_long:
            new_signal = SIZE
        # Also enter on HMA crossover with daily support
        elif daily_bullish and hma_cross_long and rsi[i] > 30:
            new_signal = SIZE
        # Long on Supertrend flip with volume
        elif st_direction[i] == 1 and st_direction[i-1] == -1 and vol_confirm:
            new_signal = SIZE
        
        # Short entry: daily bearish + HMA trend + Supertrend + RSI ok
        elif daily_bearish and hma_trend_short and st_short and rsi_pullback_short:
            new_signal = -SIZE
        # Also enter on HMA crossover with daily resistance
        elif daily_bearish and hma_cross_short and rsi[i] < 70:
            new_signal = -SIZE
        # Short on Supertrend flip with volume
        elif st_direction[i] == -1 and st_direction[i-1] == 1 and vol_confirm:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs - update trailing stop
            else:
                trailing_stop[i] = max(trailing_stop[i-1] if i > 0 else 0, close[i] - 2.5 * atr[i])
                if close[i] < trailing_stop[i] and trailing_stop[i] > 0:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] > entry_price[i-1] + 3.0 * atr[i] and new_signal == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts
            else:
                trailing_stop[i] = min(trailing_stop[i-1] if i > 0 else 999999, close[i] + 2.5 * atr[i])
                if close[i] > trailing_stop[i] and trailing_stop[i] < 999999:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] < entry_price[i-1] - 3.0 * atr[i] and new_signal == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            trailing_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                trailing_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            else:
                trailing_stop[i] = trailing_stop[i-1] if i > 0 else trailing_stop[i]
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals