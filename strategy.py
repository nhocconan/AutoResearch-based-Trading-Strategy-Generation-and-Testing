#!/usr/bin/env python3
"""
Experiment #268: 4h Fisher Transform + Daily HMA Trend + ADX Momentum Filter
Hypothesis: Fisher Transform excels at identifying reversal points in bear/range markets
(2022 crash, 2025 sideways). Combined with Daily HMA for trend bias and ADX to filter
low-momentum periods, this should generate trades during both trending and ranging phases.
Unlike previous failed 4h strategies, this uses simpler entry conditions to ensure
sufficient trade frequency. Fisher crosses -1.5 for long, +1.5 for short. Position
sizing: 0.25 entry, 0.125 half at 2R. Stoploss: 2.5*ATR trailing. Target: Beat Sharpe=0.499.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_daily_hma_adx_momentum_atr_v1"
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
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - identifies turning points.
    Formula: Fisher = 0.5 * ln((1+X)/(1-X)) where X = 0.67 * (price - lowest)/(highest - lowest) + 0.33
    """
    close_s = pd.Series(close)
    highest = close_s.rolling(window=period, min_periods=period).max().values
    lowest = close_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize price to -1 to +1 range
    range_val = highest - lowest
    range_val = np.where(range_val == 0, 1e-10, range_val)  # avoid division by zero
    x = 0.67 * (close - lowest) / range_val + 0.33
    x = np.clip(x, -0.999, 0.999)  # ensure valid ln input
    
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    fisher = pd.Series(fisher).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di.values, minus_di.values

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher(close, 9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Track previous values for crossover detection
    prev_fisher = np.roll(fisher, 1)
    prev_fisher[0] = fisher[0]
    prev_plus_di = np.roll(plus_di, 1)
    prev_plus_di[0] = plus_di[0]
    prev_minus_di = np.roll(minus_di, 1)
    prev_minus_di[0] = minus_di[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
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
        
        # Fisher Transform signals (reversal detection)
        fisher_cross_up = prev_fisher[i] <= -1.5 and fisher[i] > -1.5
        fisher_cross_down = prev_fisher[i] >= 1.5 and fisher[i] < 1.5
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # ADX momentum filter (only trade when trend exists)
        adx_strong = adx[i] > 20  # minimum trend strength
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = minus_di[i] > plus_di[i]
        
        # RSI confirmation (loose filter to ensure trades)
        rsi_bullish = rsi[i] > 35
        rsi_bearish = rsi[i] < 65
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Fisher cross up from oversold with trend confirmation
        if fisher_cross_up:
            if daily_bullish and adx_strong and rsi_bullish:
                new_signal = SIZE_ENTRY
            elif weekly_bullish and di_bullish:
                new_signal = SIZE_ENTRY
        
        # Fisher oversold bounce in uptrend
        elif fisher_oversold and daily_bullish:
            if fisher[i] > prev_fisher[i] and adx_strong:
                new_signal = SIZE_ENTRY
        
        # DI crossover with Fisher confirmation
        elif di_bullish and prev_plus_di[i] <= prev_minus_di[i]:
            if daily_bullish and fisher[i] > -1.0:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Fisher cross down from overbought with trend confirmation
        if fisher_cross_down:
            if daily_bearish and adx_strong and rsi_bearish:
                new_signal = -SIZE_ENTRY
            elif weekly_bearish and di_bearish:
                new_signal = -SIZE_ENTRY
        
        # Fisher overbought rejection in downtrend
        elif fisher_overbought and daily_bearish:
            if fisher[i] < prev_fisher[i] and adx_strong:
                new_signal = -SIZE_ENTRY
        
        # DI crossover with Fisher confirmation
        elif di_bearish and prev_minus_di[i] <= prev_plus_di[i]:
            if daily_bearish and fisher[i] < 1.0:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
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
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
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
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
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
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals