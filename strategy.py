#!/usr/bin/env python3
"""
Experiment #012: 1d KAMA Trend + Weekly Regime + RSI Pullback
Hypothesis: Daily timeframe reduces noise significantly vs intraday.
KAMA (Kaufman Adaptive MA) adapts to market efficiency - slower in chop, faster in trends.
Weekly KAMA provides major regime filter (bull/bear market identification).
RSI pullback entries within trend direction reduce whipsaw.
ATR-based stoploss (2.5x) protects against crashes.
Position sizing at 0.25-0.30 with discrete levels to minimize fee churn.
Relaxed entry conditions to ensure ≥10 trades/symbol on train data.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_weekly_rsi_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    direction = np.abs(close_s - close_s.shift(period))
    volatility = close_s.rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(np.abs(x - x.shift(1)))
    )
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_sma(close, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load weekly HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    kama_1w = calculate_kama(df_1w['close'].values, 10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate daily indicators
    atr = calculate_atr(high, low, close, 14)
    kama_fast = calculate_kama(close, 5)
    kama_slow = calculate_kama(close, 20)
    rsi = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=1.0)
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trailing_stop = np.zeros(n)
    
    for i in range(200, n):
        # Weekly trend filter (major regime)
        weekly_bullish = kama_1w_aligned[i] > 0 and close[i] > kama_1w_aligned[i]
        weekly_bearish = kama_1w_aligned[i] > 0 and close[i] < kama_1w_aligned[i]
        
        # Daily KAMA trend
        kama_trend_long = kama_fast[i] > kama_slow[i]
        kama_trend_short = kama_fast[i] < kama_slow[i]
        
        # KAMA crossover signals
        kama_cross_long = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        kama_cross_short = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # Price vs SMA200 filter
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # RSI pullback entries (relaxed for more trades)
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 60
        rsi_pullback_short = rsi[i] > 40 and rsi[i] < 65
        rsi_rising = rsi[i] > rsi[i-3] if i > 3 else True
        rsi_falling = rsi[i] < rsi[i-3] if i > 3 else True
        
        # Donchian breakout
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # Entry logic - multiple entry paths to ensure trades
        new_signal = 0.0
        
        # Long entry path 1: Weekly bullish + KAMA trend + RSI pullback
        if weekly_bullish and kama_trend_long and rsi_pullback_long and above_sma200:
            new_signal = SIZE
        # Long entry path 2: KAMA crossover with weekly support
        elif weekly_bullish and kama_cross_long and rsi[i] > 30:
            new_signal = SIZE
        # Long entry path 3: Donchian breakout with trend
        elif weekly_bullish and donchian_breakout_long and kama_trend_long:
            new_signal = SIZE
        # Long entry path 4: RSI oversold bounce in uptrend
        elif weekly_bullish and rsi[i] < 35 and rsi_rising and kama_trend_long:
            new_signal = SIZE
        
        # Short entry path 1: Weekly bearish + KAMA trend + RSI pullback
        elif weekly_bearish and kama_trend_short and rsi_pullback_short and below_sma200:
            new_signal = -SIZE
        # Short entry path 2: KAMA crossover with weekly resistance
        elif weekly_bearish and kama_cross_short and rsi[i] < 70:
            new_signal = -SIZE
        # Short entry path 3: Donchian breakdown with trend
        elif weekly_bearish and donchian_breakout_short and kama_trend_short:
            new_signal = -SIZE
        # Short entry path 4: RSI overbought drop in downtrend
        elif weekly_bearish and rsi[i] > 65 and rsi_falling and kama_trend_short:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs
            else:
                prev_trail = trailing_stop[i-1] if i > 0 else 0
                trailing_stop[i] = max(prev_trail, close[i] - 2.5 * atr[i])
                if close[i] < trailing_stop[i] and trailing_stop[i] > 0:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] > entry_price[i-1] + 3.0 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts
            else:
                prev_trail = trailing_stop[i-1] if i > 0 else 999999
                trailing_stop[i] = min(prev_trail, close[i] + 2.5 * atr[i])
                if close[i] > trailing_stop[i] and trailing_stop[i] < 999999:
                    new_signal = 0.0
                # Take partial profit at 3R
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
                trailing_stop[i] = trailing_stop[i-1] if i > 0 else trailing_stop[i]
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals