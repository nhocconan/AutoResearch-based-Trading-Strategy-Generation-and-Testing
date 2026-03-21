#!/usr/bin/env python3
"""
Experiment #002: 30m Supertrend + 4h HMA Trend Filter + RSI Pullback
Hypothesis: 4h HMA defines primary trend, 30m Supertrend gives entry timing,
RSI pullback entries reduce false signals. ATR stoploss protects capital.
This should work across BTC/ETH/SOL by adapting to regime (trend vs range).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_30m_v1"
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

def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper[0]
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            if direction[i-1] == 1:
                supertrend[i] = max(supertrend[i], lower[i])
            else:
                supertrend[i] = min(supertrend[i], upper[i])
    
    return supertrend, direction

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_bollinger_bw(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width for regime detection."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bw = (upper - lower) / sma
    bw = np.nan_to_num(bw, nan=0.0)
    return bw, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, atr, 3.0)
    rsi = calculate_rsi(close, 14)
    bb_bw, bb_sma = calculate_bollinger_bw(close, 20, 2.0)
    
    # Calculate BB percentile for regime detection
    bb_percentile = np.zeros(n)
    for i in range(50, n):
        bb_percentile[i] = np.percentile(bb_bw[max(0,i-100):i+1], 50)
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    highest_price = np.zeros(n)
    lowest_price = np.zeros(n)
    
    for i in range(50, n):
        # 4h trend filter
        hma_trend = 1 if hma_4h_aligned[i] > 0 and close[i] > hma_4h_aligned[i] else -1
        
        # Supertrend direction
        st_trend = st_direction[i]
        
        # RSI condition
        rsi_long = rsi[i] < 60  # Not overbought for long
        rsi_short = rsi[i] > 40  # Not oversold for short
        
        # Regime detection (low BW = range, high BW = trend)
        is_trending = bb_bw[i] > bb_percentile[i] * 1.2
        
        # Entry logic
        new_signal = 0.0
        
        if hma_trend > 0 and st_trend > 0 and rsi_long:
            # Bullish: 4h uptrend + Supertrend bullish + RSI not overbought
            new_signal = SIZE
        elif hma_trend < 0 and st_trend < 0 and rsi_short:
            # Bearish: 4h downtrend + Supertrend bearish + RSI not oversold
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6)
        if position_side > 0 and entry_price[i-1] > 0:
            if close[i] < entry_price[i-1] - 2.0 * atr[i]:
                new_signal = 0.0  # Stoploss hit
            elif close[i] > highest_price[i-1]:
                # Trail stop for longs
                if close[i] > entry_price[i-1] + 2.0 * atr[i]:
                    new_signal = HALF_SIZE  # Take partial profit
        
        if position_side < 0 and entry_price[i-1] > 0:
            if close[i] > entry_price[i-1] + 2.0 * atr[i]:
                new_signal = 0.0  # Stoploss hit
            elif close[i] < lowest_price[i-1]:
                # Trail stop for shorts
                if close[i] < entry_price[i-1] - 2.0 * atr[i]:
                    new_signal = -HALF_SIZE  # Take partial profit
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            highest_price[i] = close[i]
            lowest_price[i] = close[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
            highest_price[i] = max(highest_price[i-1], close[i])
            lowest_price[i] = min(lowest_price[i-1], close[i])
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            highest_price[i] = highest_price[i-1] if i > 0 else close[i]
            lowest_price[i] = lowest_price[i-1] if i > 0 else close[i]
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals