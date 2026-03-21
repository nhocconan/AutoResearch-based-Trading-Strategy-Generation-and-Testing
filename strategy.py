#!/usr/bin/env python3
"""
Experiment #010: 4h KAMA Adaptive Trend + Daily Regime + Volume Breakout
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency -
fast in trends, slow in chop. Combined with daily regime filter and volume
breakout confirmation, this should reduce whipsaws in bear/range markets (2022, 2025)
while capturing multi-day swings on 4h timeframe.
Position sizing: 0.25 discrete, stoploss at 2.5*ATR, take profit at 3R.
Relaxed entry conditions to ensure ≥10 trades/symbol on train data.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_daily_volbreakout_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    Adapts to market efficiency - fast in trends, slow in chop.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.abs(close[0:period] - close[0])
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=period, min_periods=period).sum().values
    volatility[0:period] = np.abs(close[0:period] - close[0])
    
    er = np.zeros(len(close))
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_volume_breakout(volume, lookback=20):
    """
    Larry Williams volume breakout signal.
    Returns ratio of current volume to average volume.
    """
    vol_sma = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().values
    vol_ratio = volume / vol_sma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    kama_1d = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama_fast = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, period=21, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_breakout(volume, 20)
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trailing_stop = np.zeros(n)
    
    for i in range(100, n):
        # Daily trend filter (major regime)
        daily_bullish = kama_1d_aligned[i] > 0 and close[i] > kama_1d_aligned[i]
        daily_bearish = kama_1d_aligned[i] > 0 and close[i] < kama_1d_aligned[i]
        
        # 4h KAMA trend
        kama_trend_long = kama_fast[i] > kama_slow[i]
        kama_trend_short = kama_fast[i] < kama_slow[i]
        
        # KAMA crossover signals
        kama_cross_long = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        kama_cross_short = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # Volume breakout confirmation (Larry Williams style)
        vol_breakout_long = vol_ratio[i] > 1.2  # 20% above average
        vol_breakout_short = vol_ratio[i] > 1.2
        
        # RSI filter (avoid extreme overbought/oversold for entries)
        rsi_ok_long = rsi[i] > 30 and rsi[i] < 75
        rsi_ok_short = rsi[i] > 25 and rsi[i] < 70
        
        # RSI momentum
        rsi_momentum_long = rsi[i] > rsi[i-3] if i > 3 else True
        rsi_momentum_short = rsi[i] < rsi[i-3] if i > 3 else True
        
        # Entry logic - multiple paths to ensure trades
        new_signal = 0.0
        
        # Long entry path 1: Daily bullish + KAMA trend + volume
        if daily_bullish and kama_trend_long and vol_breakout_long and rsi_ok_long:
            new_signal = SIZE
        # Long entry path 2: KAMA crossover with volume confirmation
        elif kama_cross_long and vol_ratio[i] > 1.5 and rsi[i] > 40:
            new_signal = SIZE
        # Long entry path 3: Price above both KAMAs + RSI rising
        elif close[i] > kama_fast[i] and close[i] > kama_slow[i] and rsi_momentum_long and vol_ratio[i] > 1.0:
            new_signal = SIZE
        
        # Short entry path 1: Daily bearish + KAMA trend + volume
        elif daily_bearish and kama_trend_short and vol_breakout_short and rsi_ok_short:
            new_signal = -SIZE
        # Short entry path 2: KAMA crossover with volume confirmation
        elif kama_cross_short and vol_ratio[i] > 1.5 and rsi[i] < 60:
            new_signal = -SIZE
        # Short entry path 3: Price below both KAMAs + RSI falling
        elif close[i] < kama_fast[i] and close[i] < kama_slow[i] and rsi_momentum_short and vol_ratio[i] > 1.0:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for longs
                current_trail = close[i] - 2.5 * atr[i]
                trailing_stop[i] = max(trailing_stop[i-1] if i > 0 else 0, current_trail)
                if close[i] < trailing_stop[i] and trailing_stop[i] > 0:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] > entry_price[i-1] + 3.0 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for shorts
                current_trail = close[i] + 2.5 * atr[i]
                trailing_stop[i] = min(trailing_stop[i-1] if i > 0 else 999999, current_trail)
                if close[i] > trailing_stop[i] and trailing_stop[i] < 999999:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] < entry_price[i-1] - 3.0 * atr[i] and signals[i-1] == -SIZE:
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
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0
        
        signals[i] = new_signal
    
    return signals