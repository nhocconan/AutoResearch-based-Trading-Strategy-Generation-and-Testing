#!/usr/bin/env python3
"""
Experiment #016: 4h KAMA Trend + Daily/Weekly Regime Filter + Supertrend Entry
Hypothesis: 4h timeframe balances noise reduction with trade frequency.
KAMA (Kaufman Adaptive MA) adapts to volatility - fast in trends, slow in ranges.
Daily HMA provides major trend regime filter. Weekly HMA adds ultra-long-term bias.
Supertrend gives precise entry timing with ATR-based stops.
Relaxed entry conditions ensure ≥10 trades/symbol while maintaining quality.
Position sizing capped at 0.30 with discrete levels to minimize fee churn.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_daily_weekly_supertrend_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    direction = np.abs(close - np.roll(close, period))
    direction[:period] = np.abs(close[:period] - close[0])
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=period, min_periods=period).sum().values
    volatility[0] = direction[0]
    
    er = np.zeros(len(close))
    mask = volatility > 0
    er[mask] = direction[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    direction = np.ones(len(close))
    
    supertrend[0] = upper[0]
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        else:
            supertrend[i] = upper[i]
            direction[i] = -1
    
    return supertrend, direction

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama_fast = calculate_kama(close, 10, 2, 30)
    kama_slow = calculate_kama(close, 30, 2, 30)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 2.5)
    adx = calculate_adx(high, low, close, 14)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.mean(volume))
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trailing_stop = np.zeros(n)
    
    for i in range(100, n):
        # Weekly trend filter (ultra-long-term bias)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Daily trend filter (major regime)
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # 4h KAMA trend
        kama_trend_long = kama_fast[i] > kama_slow[i]
        kama_trend_short = kama_fast[i] < kama_slow[i]
        
        # KAMA crossover signals
        kama_cross_long = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        kama_cross_short = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # Supertrend confirmation
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip signals
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # ADX trend strength (relaxed threshold for more trades)
        adx_strong = adx[i] > 18  # Lower threshold for 4h
        
        # RSI filter (not extreme, relaxed)
        rsi_ok_long = rsi[i] > 30 and rsi[i] < 75
        rsi_ok_short = rsi[i] > 25 and rsi[i] < 70
        rsi_rising = rsi[i] > rsi[i-3] if i > 3 else True
        rsi_falling = rsi[i] < rsi[i-3] if i > 3 else True
        
        # Volume confirmation (relaxed)
        vol_confirm = volume[i] > vol_sma[i] * 0.6 if vol_sma[i] > 0 else True
        
        # Entry logic - multiple paths to ensure trades
        new_signal = 0.0
        
        # Path 1: Full trend alignment (highest confidence)
        if weekly_bullish and daily_bullish and kama_trend_long and st_long and adx_strong:
            new_signal = SIZE
        elif weekly_bearish and daily_bearish and kama_trend_short and st_short and adx_strong:
            new_signal = -SIZE
        
        # Path 2: KAMA crossover with daily support (good for entries)
        elif daily_bullish and kama_cross_long and rsi_ok_long and vol_confirm:
            new_signal = SIZE
        elif daily_bearish and kama_cross_short and rsi_ok_short and vol_confirm:
            new_signal = -SIZE
        
        # Path 3: Supertrend flip with volume (catches reversals)
        elif st_flip_long and vol_confirm and rsi_ok_long:
            new_signal = SIZE
        elif st_flip_short and vol_confirm and rsi_ok_short:
            new_signal = -SIZE
        
        # Path 4: Daily trend + KAMA trend (simpler, more trades)
        elif daily_bullish and kama_trend_long and st_long:
            new_signal = SIZE
        elif daily_bearish and kama_trend_short and st_short:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0
            else:
                trailing_stop[i] = max(trailing_stop[i-1] if i > 0 else 0, close[i] - 2.5 * atr[i])
                if close[i] < trailing_stop[i] and trailing_stop[i] > 0:
                    new_signal = 0.0
                elif close[i] > entry_price[i-1] + 3.0 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0
            else:
                trailing_stop[i] = min(trailing_stop[i-1] if i > 0 else 999999, close[i] + 2.5 * atr[i])
                if close[i] > trailing_stop[i] and trailing_stop[i] < 999999:
                    new_signal = 0.0
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