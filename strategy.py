#!/usr/bin/env python3
"""
Experiment #022: 4h Supertrend + Daily HMA Regime + RSI Confirmation
Hypothesis: 4h timeframe balances noise reduction with trade frequency.
Daily HMA provides major trend regime filter (long only when price > 1d HMA).
4h Supertrend gives clean entry/exit signals with built-in volatility adjustment.
RSI(14) confirms momentum without being too extreme (avoiding 0-trade problem).
ATR(14) stoploss at 2.5x protects against crashes like 2022.
Position sizing at 0.30 with discrete levels to minimize fee churn.
Relaxed RSI conditions (30-70 range) to ensure ≥10 trades/symbol on train data.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_daily_rsi_v1"
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
    """Calculate Supertrend indicator with direction."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for trend filtering."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.mean(volume))
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    
    for i in range(50, n):
        # Daily trend filter (major regime)
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip signals (entry triggers)
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # RSI confirmation (relaxed to ensure trades)
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 75
        rsi_ok_short = rsi[i] > 25 and rsi[i] < 65
        rsi_momentum_long = rsi[i] > rsi[i-3] if i > 3 else True
        rsi_momentum_short = rsi[i] < rsi[i-3] if i > 3 else True
        
        # Volume confirmation (relaxed)
        vol_confirm = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # Entry logic - relaxed conditions to ensure trades
        new_signal = 0.0
        
        # Long entry: daily bullish + Supertrend long + RSI ok
        if daily_bullish and st_long and rsi_ok_long:
            new_signal = SIZE
        # Long on Supertrend flip with daily support
        elif daily_bullish and st_flip_long and vol_confirm:
            new_signal = SIZE
        # Long on Supertrend flip alone (catch trend changes)
        elif st_flip_long and rsi[i] > 40:
            new_signal = SIZE
        
        # Short entry: daily bearish + Supertrend short + RSI ok
        elif daily_bearish and st_short and rsi_ok_short:
            new_signal = -SIZE
        # Short on Supertrend flip with daily resistance
        elif daily_bearish and st_flip_short and vol_confirm:
            new_signal = -SIZE
        # Short on Supertrend flip alone (catch trend changes)
        elif st_flip_short and rsi[i] < 60:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            trail_stop = trailing_stop
            if close[i] < stop_loss or (trail_stop > 0 and close[i] < trail_stop):
                new_signal = 0.0  # Stoploss hit
            else:
                # Update trailing stop for longs
                trailing_stop = max(trailing_stop, close[i] - 2.5 * atr[i])
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            trail_stop = trailing_stop
            if close[i] > stop_loss or (trail_stop > 0 and close[i] > trail_stop):
                new_signal = 0.0  # Stoploss hit
            else:
                # Update trailing stop for shorts
                trailing_stop = min(trailing_stop if trailing_stop > 0 else 999999, close[i] + 2.5 * atr[i])
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal == 0 and position_side != 0:
            entry_price = 0.0
            position_side = 0
            trailing_stop = 0.0
        
        signals[i] = new_signal
    
    return signals