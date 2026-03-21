#!/usr/bin/env python3
"""
Experiment #019: 15m Supertrend + 4h HMA Regime Filter + Volume Confirmation
Hypothesis: 15m Supertrend captures intraday trends, but needs strong 4h regime filter
to avoid whipsaws that destroyed previous 15m strategies. 4h HMA slope determines
bull/bear regime. Only trade Supertrend signals aligned with 4h trend direction.
Volume confirmation filters false breakouts. ATR stoploss at 2.0x for faster exits.
Position sizing: 0.25 discrete levels to minimize fee churn while maintaining exposure.
Relaxed entry conditions to ensure ≥10 trades/symbol (learned from 0-trade failures).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_vol_v1"
timeframe = "15m"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator with direction."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))
    
    supertrend[0] = lower[0]
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        else:
            supertrend[i] = upper[i]
            direction[i] = -1
    
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
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.mean(volume))
    
    # 4h HMA slope (trend strength)
    hma_4h_slope = np.zeros(n)
    for i in range(5, n):
        hma_4h_slope[i] = hma_4h_aligned[i] - hma_4h_aligned[i-5]
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.125
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trailing_stop = np.zeros(n)
    max_price = np.zeros(n)
    min_price = np.zeros(n)
    
    for i in range(100, n):
        # 4h regime filter (CRITICAL - only trade with HTF trend)
        hma_4h_bullish = hma_4h_aligned[i] > 0 and close[i] > hma_4h_aligned[i] and hma_4h_slope[i] > 0
        hma_4h_bearish = hma_4h_aligned[i] > 0 and close[i] < hma_4h_aligned[i] and hma_4h_slope[i] < 0
        
        # Supertrend signals
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip detection (entry trigger)
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # Volume confirmation (relaxed - 80% of SMA)
        vol_confirm = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # RSI filter (avoid extremes for trend following)
        rsi_ok_long = rsi[i] > 30 and rsi[i] < 80
        rsi_ok_short = rsi[i] > 20 and rsi[i] < 70
        
        # Entry logic - relaxed to ensure trades
        new_signal = 0.0
        
        # Long: 4h bullish + Supertrend long + volume + RSI ok
        if hma_4h_bullish and st_long and vol_confirm and rsi_ok_long:
            new_signal = SIZE
        # Long on Supertrend flip with 4h support
        elif hma_4h_bullish and st_flip_long and rsi[i] > 40:
            new_signal = SIZE
        # Long on pullback in 4h uptrend (RSI dip)
        elif hma_4h_bullish and st_long and rsi[i] > 40 and rsi[i] < rsi[i-3]:
            new_signal = SIZE
        
        # Short: 4h bearish + Supertrend short + volume + RSI ok
        elif hma_4h_bearish and st_short and vol_confirm and rsi_ok_short:
            new_signal = -SIZE
        # Short on Supertrend flip with 4h resistance
        elif hma_4h_bearish and st_flip_short and rsi[i] < 60:
            new_signal = -SIZE
        # Short on rally in 4h downtrend (RSI spike)
        elif hma_4h_bearish and st_short and rsi[i] < 60 and rsi[i] > rsi[i-3]:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based at 2.0x
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.0 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs
            else:
                max_price[i] = max(max_price[i-1] if i > 0 else 0, close[i])
                trailing_stop[i] = max(trailing_stop[i-1] if i > 0 else 0, max_price[i] - 2.0 * atr[i])
                if trailing_stop[i] > 0 and close[i] < trailing_stop[i]:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] > entry_price[i-1] + 2.5 * atr[i] and new_signal == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.0 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts
            else:
                min_price[i] = min(min_price[i-1] if i > 0 else 999999, close[i])
                trailing_stop[i] = min(trailing_stop[i-1] if i > 0 else 999999, min_price[i] + 2.0 * atr[i])
                if trailing_stop[i] < 999999 and close[i] > trailing_stop[i]:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] < entry_price[i-1] - 2.5 * atr[i] and new_signal == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            max_price[i] = close[i] if position_side > 0 else max_price[i-1] if i > 0 else close[i]
            min_price[i] = close[i] if position_side < 0 else min_price[i-1] if i > 0 else close[i]
            trailing_stop[i] = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                max_price[i] = close[i] if position_side > 0 else max_price[i-1] if i > 0 else close[i]
                min_price[i] = close[i] if position_side < 0 else min_price[i-1] if i > 0 else close[i]
                trailing_stop[i] = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            else:
                max_price[i] = max_price[i-1] if i > 0 else close[i]
                min_price[i] = min_price[i-1] if i > 0 else close[i]
                trailing_stop[i] = trailing_stop[i-1] if i > 0 else trailing_stop[i]
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            max_price[i] = max_price[i-1] if i > 0 else 0
            min_price[i] = min_price[i-1] if i > 0 else 0
            trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0
        
        signals[i] = new_signal
    
    return signals