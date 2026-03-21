#!/usr/bin/env python3
"""
Experiment #064: 4h KAMA with Daily HMA Trend Filter + RSI Momentum
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility -
faster response in trends, slower in ranges. Combined with daily HMA for trend
direction and RSI momentum filter. Simpler entry conditions than previous attempts
to ensure sufficient trade generation across all symbols (BTC/ETH/SOL).
4h timeframe balances noise reduction with trade frequency. Position sizing: 0.25
entry, reduce to 0.12 at 2R profit, stoploss at 2.5*ATR for wider breathing room.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_daily_hma_rsi_momentum_v1"
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
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.abs(close[0:period] - close[0])
    
    volatility = np.zeros(len(close))
    for i in range(period, len(close)):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    
    er = np.zeros(len(close))
    er[period:] = np.where(volatility[period:] > 0, change[period:] / volatility[period:], 0)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if i < period:
            kama[i] = close_s.iloc[:i+1].mean()
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_momentum(close, period=10):
    """Calculate rate of change momentum."""
    momentum = np.zeros(len(close))
    momentum[period:] = (close[period:] - close[:-period]) / close[:-period] * 100
    return momentum

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
    kama_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_20 = calculate_kama(close, period=20, fast_period=2, slow_period=30)
    momentum = calculate_momentum(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    
    for i in range(100, n):
        # Daily trend filter (HTF) - use HMA slope and price position
        daily_bullish = (hma_1d_aligned[i] > 0) and (close[i] > hma_1d_aligned[i])
        daily_bearish = (hma_1d_aligned[i] > 0) and (close[i] < hma_1d_aligned[i])
        
        # KAMA trend direction
        kama_bullish = kama_10[i] > kama_20[i]
        kama_bearish = kama_10[i] < kama_20[i]
        
        # KAMA flip signals
        kama_flip_long = (i > 0) and (kama_10[i] > kama_20[i]) and (kama_10[i-1] <= kama_20[i-1])
        kama_flip_short = (i > 0) and (kama_10[i] < kama_20[i]) and (kama_10[i-1] >= kama_20[i-1])
        
        # RSI momentum filter (loose thresholds to ensure trades)
        rsi_long = rsi[i] > 40  # Not oversold
        rsi_short = rsi[i] < 60  # Not overbought
        rsi_rising = (i > 2) and (rsi[i] > rsi[i-2])
        rsi_falling = (i > 2) and (rsi[i] < rsi[i-2])
        
        # Momentum confirmation
        mom_positive = momentum[i] > 0
        mom_negative = momentum[i] < 0
        
        new_signal = 0.0
        
        # LONG ENTRY: KAMA flip + Daily bullish OR KAMA trend + RSI momentum
        if kama_flip_long and daily_bullish:
            new_signal = SIZE_ENTRY
        elif kama_bullish and daily_bullish and rsi_long and rsi_rising:
            new_signal = SIZE_ENTRY
        elif kama_bullish and daily_bullish and mom_positive:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: KAMA flip + Daily bearish OR KAMA trend + RSI momentum
        if kama_flip_short and daily_bearish:
            new_signal = -SIZE_ENTRY
        elif kama_bearish and daily_bearish and rsi_short and rsi_falling:
            new_signal = -SIZE_ENTRY
        elif kama_bearish and daily_bearish and mom_negative:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = close[i] - entry_price
                    risk = 2.5 * atr[int(i)] if i > 0 else atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = SIZE_HALF
                        position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = entry_price - close[i]
                    risk = 2.5 * atr[int(i)] if i > 0 else atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = -SIZE_HALF
                        position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals