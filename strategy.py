#!/usr/bin/env python3
"""
Experiment #023: 12h Supertrend + Daily Regime + RSI Pullback v2
Hypothesis: 12h timeframe balances noise reduction with trade frequency.
Daily HMA provides major trend regime (bull/bear market filter).
Supertrend(10,3) gives clear trend direction with ATR-based stops.
RSI pullback (35-65 range) enters on dips within trend, not extremes.
Volume confirmation reduces false signals.
ATR trailing stop (2.5x) protects capital during crashes like 2022.
Position sizing: 0.25 discrete levels to minimize fee churn.
Relaxed entry conditions to ensure ≥10 trades/symbol on 4-year train data.
FIXED: Proper position tracking with scalars, not arrays.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_supertrend_daily_rsi_v2"
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
    """Calculate Hull Moving Average."""
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
    direction = np.ones(len(close))
    
    supertrend[0] = upper[0]
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
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # Volume SMA
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.mean(volume))
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking (scalars, not arrays)
    in_position = False
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    
    for i in range(100, n):
        new_signal = 0.0
        
        # Daily regime filter
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # RSI pullback (relaxed range for more trades)
        rsi_neutral = 35 < rsi[i] < 65
        
        # Volume confirmation (relaxed)
        vol_ok = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # Supertrend flip signals (more frequent entries)
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # Long entry: daily bullish + supertrend long + RSI ok
        if daily_bullish and st_long and rsi_neutral and vol_ok:
            if not in_position or position_side < 0:
                new_signal = SIZE
        
        # Long on Supertrend flip (catch trend changes)
        elif st_flip_long and vol_ok:
            if not in_position or position_side < 0:
                new_signal = SIZE
        
        # Short entry: daily bearish + supertrend short + RSI ok
        elif daily_bearish and st_short and rsi_neutral and vol_ok:
            if not in_position or position_side > 0:
                new_signal = -SIZE
        
        # Short on Supertrend flip
        elif st_flip_short and vol_ok:
            if not in_position or position_side > 0:
                new_signal = -SIZE
        
        # Stoploss logic (Rule 6)
        if in_position and position_side > 0:
            # Long stoploss
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0
            else:
                # Trail stop
                trailing_stop = max(trailing_stop, close[i] - 2.5 * atr[i])
                if close[i] < trailing_stop:
                    new_signal = 0.0
        
        elif in_position and position_side < 0:
            # Short stoploss
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0
            else:
                # Trail stop
                trailing_stop = min(trailing_stop, close[i] + 2.5 * atr[i])
                if close[i] > trailing_stop:
                    new_signal = 0.0
        
        # Update position tracking
        if new_signal != 0:
            if not in_position or np.sign(new_signal) != position_side:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal == 0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
        
        signals[i] = new_signal
    
    return signals