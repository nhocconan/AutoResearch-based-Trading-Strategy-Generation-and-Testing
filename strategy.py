#!/usr/bin/env python3
"""
Experiment #030: Daily Donchian Breakout + Weekly HMA Regime + Volume Confirmation
Hypothesis: Donchian Channel breakouts (Turtle Trading style) work well on daily timeframe
for capturing multi-week trends while avoiding intraday noise. Weekly HMA provides major
regime filter to avoid counter-trend breakouts. Volume confirmation filters false breakouts.
Multiple entry triggers (breakout, pullback, continuation) ensure ≥10 trades per symbol.
Position sizing 0.30 with 2.5x ATR stoploss protects against 2022-style crashes.
This is DIFFERENT from #024's Supertrend approach - Donchian is pure price breakout system.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_hma_vol_v1"
timeframe = "1d"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (Turtle Trading breakout system)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

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
    
    # Load weekly HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate daily indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    # Daily HMA for trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # Volume SMA and volatility for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.nanmean(volume))
    vol_std = pd.Series(volume).rolling(window=20, min_periods=20).std().values
    vol_std = np.nan_to_num(vol_std, nan=np.std(volume))
    
    # Price momentum (ROC)
    roc = np.zeros(n)
    for i in range(10, n):
        roc[i] = (close[i] - close[i-10]) / close[i-10] * 100 if close[i-10] > 0 else 0
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Weekly trend filter (major regime)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 and donchian_upper[i-1] > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 and donchian_lower[i-1] > 0 else False
        
        # Donchian pullback (price near lower band in uptrend, or upper in downtrend)
        pullback_long = close[i] < donchian_mid[i] and close[i] > donchian_lower[i]
        pullback_short = close[i] > donchian_mid[i] and close[i] < donchian_upper[i]
        
        # HMA trend confirmation
        hma_trend_long = hma_21[i] > hma_50[i] and hma_21[i] > 0
        hma_trend_short = hma_21[i] < hma_50[i] and hma_21[i] > 0
        
        # RSI momentum (avoid overbought/oversold extremes for entries)
        rsi_bullish = rsi[i] > 45 and rsi[i] < 75
        rsi_bearish = rsi[i] > 25 and rsi[i] < 55
        rsi_rising = rsi[i] > rsi[i-3] if i > 3 else True
        rsi_falling = rsi[i] < rsi[i-3] if i > 3 else True
        
        # Volume confirmation (spike above average)
        vol_spike = volume[i] > vol_sma[i] * 1.2 if vol_sma[i] > 0 else True
        vol_confirm = volume[i] > vol_sma[i] * 0.9 if vol_sma[i] > 0 else True
        
        # Price momentum confirmation
        momentum_long = roc[i] > 2.0
        momentum_short = roc[i] < -2.0
        
        # Entry logic - MULTIPLE triggers to ensure trades (Rule 9)
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS (any one can trigger)
        # Trigger 1: Donchian breakout long with weekly support + volume
        if breakout_long and (weekly_bullish or hma_trend_long) and vol_spike:
            new_signal = SIZE
        # Trigger 2: Donchian breakout + HMA trend + RSI ok (classic Turtle)
        elif breakout_long and hma_trend_long and rsi_bullish:
            new_signal = SIZE
        # Trigger 3: Pullback to Donchian mid in uptrend (buy the dip)
        elif pullback_long and hma_trend_long and weekly_bullish and rsi_rising:
            new_signal = SIZE
        # Trigger 4: Weekly bullish + HMA trend + momentum (trend continuation)
        elif weekly_bullish and hma_trend_long and momentum_long and vol_confirm:
            new_signal = SIZE
        # Trigger 5: RSI rising from neutral with trend support
        elif rsi[i] > 50 and rsi_rising and hma_trend_long:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS (any one can trigger)
        # Trigger 1: Donchian breakout short with weekly resistance + volume
        if breakout_short and (weekly_bearish or hma_trend_short) and vol_spike:
            new_signal = -SIZE
        # Trigger 2: Donchian breakout + HMA trend + RSI ok (classic Turtle)
        elif breakout_short and hma_trend_short and rsi_bearish:
            new_signal = -SIZE
        # Trigger 3: Pullback to Donchian mid in downtrend (sell the rally)
        elif pullback_short and hma_trend_short and weekly_bearish and rsi_falling:
            new_signal = -SIZE
        # Trigger 4: Weekly bearish + HMA trend + momentum (trend continuation)
        elif weekly_bearish and hma_trend_short and momentum_short and vol_confirm:
            new_signal = -SIZE
        # Trigger 5: RSI falling from neutral with trend support
        elif rsi[i] < 50 and rsi_falling and hma_trend_short:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based with trailing
        if position_side > 0 and entry_price > 0:
            # Update highest price since entry for trailing
            if close[i] > highest_since_entry:
                highest_since_entry = close[i]
            
            stop_loss = entry_price - 2.5 * atr[i]
            trailing_loss = highest_since_entry - 2.5 * atr[i]
            
            if close[i] < stop_loss or close[i] < trailing_loss:
                new_signal = 0.0  # Stoploss hit
            # Take partial profit at 3R
            elif close[i] > entry_price + 3.0 * atr[i] and signals[i-1] == SIZE:
                new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            # Update lowest price since entry for trailing
            if lowest_since_entry == 0.0 or close[i] < lowest_since_entry:
                lowest_since_entry = close[i]
            
            stop_loss = entry_price + 2.5 * atr[i]
            trailing_loss = lowest_since_entry + 2.5 * atr[i]
            
            if close[i] > stop_loss or close[i] > trailing_loss:
                new_signal = 0.0  # Stoploss hit
            # Take partial profit at 3R
            elif close[i] < entry_price - 3.0 * atr[i] and signals[i-1] == -SIZE:
                new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            highest_since_entry = close[i] if position_side > 0 else 0.0
            lowest_since_entry = close[i] if position_side < 0 else 0.0
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals