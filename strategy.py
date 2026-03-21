#!/usr/bin/env python3
"""
Experiment #012: Daily Trend + 4h HMA Filter + Z-Score Pullback
Hypothesis: Daily timeframe with 4h trend filter provides better responsiveness than weekly.
Z-score pullback entries in trending markets reduce whipsaw vs pure crossover.
ATR stoploss at 2.5x protects against crashes while allowing normal volatility.
Position sizing: 0.25 base, 0.30 on strong confirmation, discrete levels to minimize churn.
This should generate 30-50 trades/year with better entry timing than EMA crossover.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_zscore_4h_trend_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion detection."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / np.where(std > 0, std, 1e-10)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate daily indicators
    atr = calculate_atr(high, low, close, 14)
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=0.0)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # 4h trend filter
        trend_bullish = hma_4h_aligned[i] > 0 and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_aligned[i] > 0 and close[i] < hma_4h_aligned[i]
        
        # Daily trend
        daily_bullish = ema_fast[i] > ema_slow[i] and close[i] > ema_50[i]
        daily_bearish = ema_fast[i] < ema_slow[i] and close[i] < ema_50[i]
        
        # Z-score pullback entry (buy dips in uptrend)
        pullback_long = zscore[i] < -0.5 and zscore[i] > -2.0 and trend_bullish
        pullback_short = zscore[i] > 0.5 and zscore[i] < 2.0 and trend_bearish
        
        # RSI momentum
        rsi_long = rsi[i] > 40 and rsi[i] < 70
        rsi_short = rsi[i] < 60 and rsi[i] > 30
        
        # Volume confirmation
        vol_ok = volume[i] > vol_sma[i] * 0.7 if vol_sma[i] > 0 else True
        
        # Entry signals - relaxed to ensure trades
        new_signal = 0.0
        
        # Long: trend + pullback OR trend + RSI momentum
        if (trend_bullish and daily_bullish and pullback_long and rsi_long and vol_ok):
            new_signal = SIZE_STRONG
        elif (trend_bullish and daily_bullish and rsi_long and rsi[i] > rsi[i-1]):
            new_signal = SIZE_BASE
        elif (daily_bullish and rsi[i] < 50 and rsi[i] > rsi[i-1]):
            new_signal = SIZE_BASE
        
        # Short: trend + pullback OR trend + RSI momentum
        if new_signal == 0:
            if (trend_bearish and daily_bearish and pullback_short and rsi_short and vol_ok):
                new_signal = -SIZE_STRONG
            elif (trend_bearish and daily_bearish and rsi_short and rsi[i] < rsi[i-1]):
                new_signal = -SIZE_BASE
            elif (daily_bearish and rsi[i] > 50 and rsi[i] < rsi[i-1]):
                new_signal = -SIZE_BASE
        
        # Stoploss logic
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0
            # Trail stop - take profit
            trailing_stop = highest_since_entry - 2.5 * atr[i]
            if close[i] < trailing_stop and close[i] > entry_price:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0
            # Trail stop - take profit
            trailing_stop = lowest_since_entry + 2.5 * atr[i]
            if close[i] > trailing_stop and close[i] < entry_price:
                new_signal = 0.0
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            highest_since_entry = close[i]
            lowest_since_entry = close[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            else:
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i])
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals