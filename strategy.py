#!/usr/bin/env python3
"""
Experiment #004: 4h Donchian Breakout + Daily Trend Filter + ATR Stop
Hypothesis: 4h timeframe captures medium-term swings better than intraday noise.
Daily EMA provides major trend direction to filter false breakouts.
Donchian Channel (20-period) breakout captures momentum moves in both directions.
ATR-based trailing stop protects against reversals (2.5*ATR).
RSI filter avoids entering at extremes (overbought/oversold).
Position sizing capped at 0.28 to limit drawdown during volatile periods.
This should generate 30-60 trades/year with better risk-adjusted returns than pure trend.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_daily_v1"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    ema_1d = calculate_ema(df_1d['close'].values, 21)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # EMA for trend confirmation on 4h
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=0.0)
    
    signals = np.zeros(n)
    SIZE = 0.28
    HALF_SIZE = 0.14
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trail_stop = np.zeros(n)
    
    for i in range(100, n):
        # Daily trend filter (major regime)
        daily_bullish = ema_1d_aligned[i] > 0 and close[i] > ema_1d_aligned[i]
        daily_bearish = ema_1d_aligned[i] > 0 and close[i] < ema_1d_aligned[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # EMA trend confirmation on 4h
        ema_trend_long = ema_fast[i] > ema_slow[i] and close[i] > ema_50[i]
        ema_trend_short = ema_fast[i] < ema_slow[i] and close[i] < ema_50[i]
        
        # RSI filter - avoid extremes
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 75
        rsi_ok_short = rsi[i] > 25 and rsi[i] < 65
        
        # ADX trend strength filter
        adx_strong = adx[i] > 20  # Trending market
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.7 if vol_sma[i] > 0 else True
        
        # Entry logic - relaxed to ensure trades
        new_signal = 0.0
        
        # Long entry: daily bullish + breakout or EMA trend + RSI ok
        if daily_bullish and breakout_long and rsi_ok_long and vol_confirm:
            new_signal = SIZE
        elif daily_bullish and ema_trend_long and rsi_ok_long and adx_strong:
            new_signal = SIZE
        # Also enter on pullback in bullish regime
        elif daily_bullish and ema_fast[i] > ema_slow[i] and rsi[i] < 45 and rsi[i] > 30:
            new_signal = SIZE
        
        # Short entry: daily bearish + breakout or EMA trend + RSI ok
        if daily_bearish and breakout_short and rsi_ok_short and vol_confirm:
            new_signal = -SIZE
        elif daily_bearish and ema_trend_short and rsi_ok_short and adx_strong:
            new_signal = -SIZE
        # Also enter on rally in bearish regime
        elif daily_bearish and ema_fast[i] < ema_slow[i] and rsi[i] > 55 and rsi[i] < 70:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based trailing stop
        if position_side > 0 and entry_price[i-1] > 0:
            # Initial stoploss
            initial_stop = entry_price[i-1] - 2.5 * atr[i]
            # Trail stop - move up as price increases
            trail_stop[i] = max(trail_stop[i-1] if i > 0 else initial_stop, close[i] - 2.5 * atr[i])
            
            if close[i] < trail_stop[i]:
                new_signal = 0.0  # Stoploss hit
            
            # Take partial profit at 3R
            if close[i] > entry_price[i-1] + 3.0 * atr[i-1] and new_signal != 0:
                new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            # Initial stoploss
            initial_stop = entry_price[i-1] + 2.5 * atr[i]
            # Trail stop - move down as price decreases
            trail_stop[i] = min(trail_stop[i-1] if i > 0 else initial_stop, close[i] + 2.5 * atr[i])
            
            if close[i] > trail_stop[i]:
                new_signal = 0.0  # Stoploss hit
            
            # Take partial profit at 3R
            if close[i] < entry_price[i-1] - 3.0 * atr[i-1] and new_signal != 0:
                new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            trail_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                trail_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            else:
                trail_stop[i] = trail_stop[i] if trail_stop[i] != 0 else (close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i])
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            trail_stop[i] = trail_stop[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals