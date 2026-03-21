#!/usr/bin/env python3
"""
Experiment #005: 12h Donchian Breakout + 1d HMA Trend + RSI Pullback Filter
Hypothesis: 12h timeframe captures medium-term trends while avoiding noise.
1d HMA provides slower trend filter to avoid whipsaws. Donchian breakouts
work well on higher TFs. RSI pullback ensures we enter on dips in uptrends.
ATR stoploss protects against 2022-style crashes. Regime-adaptive sizing.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_donchian_hma_12h_v1"
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
    """Calculate Hull Moving Average for trend direction."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel for breakout signals."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

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

def calculate_bollinger_bw(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width for regime detection."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bw = (upper - lower) / sma
    bw = np.nan_to_num(bw, nan=0.0)
    return bw

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    rsi = calculate_rsi(close, 14)
    bb_bw = calculate_bollinger_bw(close, 20, 2.0)
    
    # Calculate BB percentile for regime detection (rolling 100 bars)
    bb_percentile = np.zeros(n)
    for i in range(100, n):
        bb_percentile[i] = np.percentile(bb_bw[max(0,i-100):i+1], 50)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    HALF_LONG = 0.15
    HALF_SHORT = 0.12
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    highest_price = np.zeros(n)
    lowest_price = np.zeros(n)
    stoploss_price = np.zeros(n)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(donchian_upper[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend filter (slower, more reliable)
        hma_slope = hma_1d_aligned[i] - hma_1d_aligned[i-10] if i >= 10 else 0
        hma_trend_bull = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i] and hma_slope > 0
        hma_trend_bear = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i] and hma_slope < 0
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] and close[i-1] <= donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1] and close[i-1] >= donchian_lower[i-1]
        
        # RSI pullback filter (enter on dips in uptrend, rallies in downtrend)
        rsi_long_ok = rsi[i] > 40 and rsi[i] < 70  # Not oversold, not overbought
        rsi_short_ok = rsi[i] > 30 and rsi[i] < 60  # Not oversold, not overbought
        
        # Regime detection (low BW = range, high BW = trend)
        is_trending = bb_bw[i] > bb_percentile[i] * 1.1
        
        # Entry logic - more lenient to ensure trades
        new_signal = 0.0
        
        # Long entry: 1d uptrend + Donchian breakout + RSI confirmation
        if hma_trend_bull and breakout_long and rsi_long_ok:
            if is_trending:
                new_signal = SIZE_LONG
            else:
                new_signal = HALF_LONG  # Reduce size in range
        
        # Short entry: 1d downtrend + Donchian breakout + RSI confirmation
        elif hma_trend_bear and breakout_short and rsi_short_ok:
            if is_trending:
                new_signal = -SIZE_SHORT
            else:
                new_signal = -HALF_SHORT  # Reduce size in range
        
        # Alternative: RSI extreme mean reversion in range market
        if not is_trending and new_signal == 0:
            if rsi[i] < 25 and hma_trend_bull:
                new_signal = HALF_LONG  # Oversold bounce in uptrend
            elif rsi[i] > 75 and hma_trend_bear:
                new_signal = -HALF_SHORT  # Overbought drop in downtrend
        
        # Stoploss logic (Rule 6) - CRITICAL for drawdown control
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs - take profit
            elif close[i] > entry_price[i-1] + 3.0 * atr[i]:
                if new_signal == 0:
                    new_signal = HALF_LONG  # Hold partial
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts - take profit
            elif close[i] < entry_price[i-1] - 3.0 * atr[i]:
                if new_signal == 0:
                    new_signal = -HALF_SHORT  # Hold partial
        
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
            else:
                entry_price[i] = entry_price[i-1]
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