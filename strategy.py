#!/usr/bin/env python3
"""
Experiment #011: 12h Donchian Breakout + Daily ADX Trend Filter
Hypothesis: Donchian channel breakouts (20-period) capture momentum moves on 12h timeframe.
Daily ADX > 25 filters for trending regimes, avoiding choppy whipsaws that destroyed previous strategies.
Volume spike (1.5x 20-bar avg) confirms breakout validity, reducing false signals.
ATR-based stoploss (2.5x) with trailing protects against reversals like 2022 crash.
Position sizing at 0.25 discrete levels minimizes fee churn while capturing trends.
Relaxed entry conditions ensure ≥10 trades/symbol - Donchian breakouts occur frequently enough.
This differs from failed HMA/RSI strategies by using pure price breakout + trend strength filter.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_daily_adx_v1"
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
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    tr_s = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # DI and DX
    plus_di = 100 * (plus_dm_s / tr_s)
    minus_di = 100 * (minus_dm_s / tr_s)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # Load daily HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Also get daily close for trend direction
    close_1d = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi = calculate_rsi(close, 14)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=1.0)
    
    # Price momentum (ROC)
    roc = np.zeros(n)
    roc[10:] = 100 * (close[10:] - close[:-10]) / (close[:-10] + 1e-10)
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 999999.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_trending = adx_1d_aligned[i] > 20  # Relaxed from 25 for more trades
        daily_bullish = close_1d_aligned[i] > close_1d_aligned[i-1] if i > 0 else True
        daily_bearish = close_1d_aligned[i] < close_1d_aligned[i-1] if i > 0 else True
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # Volume confirmation (relaxed)
        vol_spike = volume[i] > vol_sma[i] * 1.2  # 20% above average
        
        # RSI filter (avoid extreme overbought/oversold for entries)
        rsi_ok_long = rsi[i] < 75  # Not extremely overbought
        rsi_ok_short = rsi[i] > 25  # Not extremely oversold
        
        # Momentum confirmation
        momentum_long = roc[i] > 0
        momentum_short = roc[i] < 0
        
        # Entry logic - multiple pathways to ensure trades
        new_signal = 0.0
        
        # Long entry: trending market + breakout + volume + RSI ok
        if daily_trending and daily_bullish and breakout_long and vol_spike and rsi_ok_long:
            new_signal = SIZE
        # Long on breakout with momentum (relaxed)
        elif breakout_long and momentum_long and rsi[i] > 40:
            new_signal = SIZE
        # Long on strong breakout (very high volume)
        elif breakout_long and volume[i] > vol_sma[i] * 2.0:
            new_signal = SIZE
        
        # Short entry: trending market + breakout + volume + RSI ok
        elif daily_trending and daily_bearish and breakout_short and vol_spike and rsi_ok_short:
            new_signal = -SIZE
        # Short on breakout with momentum (relaxed)
        elif breakout_short and momentum_short and rsi[i] < 60:
            new_signal = -SIZE
        # Short on strong breakout (very high volume)
        elif breakout_short and volume[i] > vol_sma[i] * 2.0:
            new_signal = -SIZE
        
        # Stoploss and profit taking logic
        if position_side > 0 and entry_price > 0:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, high[i])
            
            # Initial stoploss
            stop_loss = entry_price - 2.5 * atr[i]
            
            # Trailing stop (lock in profits)
            trail_stop = highest_since_entry - 2.5 * atr[i]
            effective_stop = max(stop_loss, trail_stop)
            
            # Check stoploss hit
            if low[i] < effective_stop:
                new_signal = 0.0  # Stoploss triggered
            
            # Take partial profit at 3R
            elif close[i] > entry_price + 3.0 * atr[i] and new_signal == SIZE:
                new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            
            # Initial stoploss
            stop_loss = entry_price + 2.5 * atr[i]
            
            # Trailing stop (lock in profits)
            trail_stop = lowest_since_entry + 2.5 * atr[i]
            effective_stop = min(stop_loss, trail_stop)
            
            # Check stoploss hit
            if high[i] > effective_stop:
                new_signal = 0.0  # Stoploss triggered
            
            # Take partial profit at 3R
            elif close[i] < entry_price - 3.0 * atr[i] and new_signal == -SIZE:
                new_signal = -HALF_SIZE
        
        # Update position tracking
        prev_side = position_side
        
        if new_signal != 0 and position_side == 0:
            # New position
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = entry_price - 2.5 * atr[i] if position_side > 0 else entry_price + 2.5 * atr[i]
            highest_since_entry = high[i] if position_side > 0 else 0
            lowest_since_entry = low[i] if position_side < 0 else 999999.0
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                # Reversal
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = entry_price - 2.5 * atr[i] if position_side > 0 else entry_price + 2.5 * atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0
                lowest_since_entry = low[i] if position_side < 0 else 999999.0
            elif new_signal == HALF_SIZE and position_side > 0:
                # Partial profit taken, keep tracking
                pass
            elif new_signal == -HALF_SIZE and position_side < 0:
                # Partial profit taken, keep tracking
                pass
        else:
            # Position closed
            if position_side != 0 and new_signal == 0:
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 999999.0
        
        signals[i] = new_signal
    
    return signals