#!/usr/bin/env python3
"""
Experiment #062: 30m Regime-Adaptive with 4h ADX Filter + 1d HMA Trend
Hypothesis: 30m is too noisy for pure trend following. Instead, adapt strategy based on 4h regime:
- Strong 4h trend (ADX > 25): Follow trend with Supertrend flips
- Weak 4h trend (ADX < 25): Mean reversion with RSI extremes
- 1d HMA as ultimate filter: Only long above, only short below
- Volume confirmation reduces false breakouts
- Conservative sizing: 0.25 entry, 0.125 take-profit reduce
- 2.5*ATR trailing stop for protection
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_adaptive_4h_adx_1d_hma_v1"
timeframe = "30m"
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
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (np.abs(plus_di + minus_di) + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx, plus_di, minus_di

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
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
    
    supertrend[0] = lower[0]
    direction[0] = 1
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

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean().values
    rolling_std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - rolling_mean) / (rolling_std + 1e-10)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        14
    )
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Calculate 1d indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    zscore = calculate_zscore(close, 20)
    
    # Volume SMA for confirmation
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (volume_sma + 1e-10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    
    for i in range(100, n):
        # 1d HMA ultimate trend filter
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # 4h ADX regime detection
        strong_trend_4h = adx_4h_aligned[i] > 25
        weak_trend_4h = adx_4h_aligned[i] <= 25
        
        # 4h HMA trend direction
        hma_4h_bullish = hma_4h_aligned[i] > 0 and close[i] > hma_4h_aligned[i]
        hma_4h_bearish = hma_4h_aligned[i] > 0 and close[i] < hma_4h_aligned[i]
        
        # 30m Supertrend signals
        st_flip_long = (i > 0) and (st_direction[i] == 1) and (st_direction[i-1] == -1)
        st_flip_short = (i > 0) and (st_direction[i] == -1) and (st_direction[i-1] == 1)
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # RSI extremes for mean reversion
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] > 40) and (rsi[i] < 60)
        
        # Z-score extremes
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] > 1.2
        
        new_signal = 0.0
        
        # REGIME 1: Strong 4h trend - Follow trend with Supertrend
        if strong_trend_4h and daily_bullish:
            if st_flip_long and volume_confirmed:
                new_signal = SIZE_ENTRY
            elif st_long and hma_4h_bullish and rsi_neutral:
                new_signal = SIZE_ENTRY
        
        if strong_trend_4h and daily_bearish:
            if st_flip_short and volume_confirmed:
                new_signal = -SIZE_ENTRY
            elif st_short and hma_4h_bearish and rsi_neutral:
                new_signal = -SIZE_ENTRY
        
        # REGIME 2: Weak 4h trend - Mean reversion with RSI/Z-score
        if weak_trend_4h and daily_bullish:
            if rsi_oversold and zscore_oversold:
                new_signal = SIZE_ENTRY
            elif rsi[i] < 35 and zscore[i] < -1.0:
                new_signal = SIZE_ENTRY
        
        if weak_trend_4h and daily_bearish:
            if rsi_overbought and zscore_overbought:
                new_signal = -SIZE_ENTRY
            elif rsi[i] > 65 and zscore[i] > 1.0:
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            current_stop = close[i] - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
            else:
                if not position_reduced:
                    profit = close[i] - entry_price
                    risk = 2.5 * atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = SIZE_HALF
                        position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            current_stop = close[i] + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
            else:
                if not position_reduced:
                    profit = entry_price - close[i]
                    risk = 2.5 * atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = -SIZE_HALF
                        position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            position_reduced = False
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            position_reduced = False
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals