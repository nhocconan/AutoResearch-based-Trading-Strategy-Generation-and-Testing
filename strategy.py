#!/usr/bin/env python3
"""
Experiment #308: 30m Regime-Adaptive Strategy with 4h HMA Bias + Choppiness Index
Hypothesis: 30m timeframe captures intraday swings while 4h HMA provides trend bias.
Choppiness Index (14) detects regime: CHOP>61.8=range(mean revert), CHOP<38.2=trend(follow).
In range: RSI extremes (25/75) with Bollinger mean reversion. In trend: breakout + momentum.
This adapts to 2022 crash (trend short) and 2025 bear/range (mean reversion). 
Position size 0.25 balances risk. ATR stops at 2.5*ATR. Target: Beat Sharpe=0.499.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_chop_4h_hma_rsi_boll_atr_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range > 0, price_range, 1e-10)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_supertrend(high, low, close, period=10, mult=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1] if i > 0 else close[i] > lower[i]:
            supertrend[i] = lower[i]
            direction[i] = 1
        else:
            supertrend[i] = upper[i]
            direction[i] = -1
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    boll_upper, boll_lower, boll_mid = calculate_bollinger(close, 20, 2.0)
    chop = calculate_choppiness(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(boll_upper[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias
        hma_valid = not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_valid and close[i] < hma_4h_aligned[i]
        
        # Regime detection via Choppiness Index
        is_range = chop[i] > 55.0  # Slightly lower threshold for more signals
        is_trend = chop[i] < 45.0  # Slightly higher threshold for more signals
        
        # Supertrend direction
        st_bullish = st_direction[i] > 0
        st_bearish = st_direction[i] < 0
        
        # RSI conditions
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 40 < rsi[i] < 60
        
        # Bollinger conditions
        near_lower = close[i] < boll_lower[i] * 1.005
        near_upper = close[i] > boll_upper[i] * 0.995
        below_mid = close[i] < boll_mid[i]
        above_mid = close[i] > boll_mid[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Range regime: Mean reversion at Bollinger lower + RSI oversold
        if is_range and near_lower and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Range regime: RSI < 30 + price above 4h HMA (bullish bias)
        elif is_range and rsi[i] < 30 and trend_bullish:
            new_signal = SIZE_ENTRY
        # Trend regime: Supertrend bullish + breakout above Bollinger mid
        elif is_trend and st_bullish and above_mid and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        # Trend regime: 4h HMA bullish + Supertrend flip bullish
        elif trend_bullish and st_bullish and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Simple: RSI < 35 + Supertrend bullish (catches reversals)
        elif rsi[i] < 35 and st_bullish:
            new_signal = SIZE_ENTRY
        # Simple: Price at Bollinger lower + 4h HMA bullish
        elif near_lower and trend_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Range regime: Mean reversion at Bollinger upper + RSI overbought
        if is_range and near_upper and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Range regime: RSI > 70 + price below 4h HMA (bearish bias)
        elif is_range and rsi[i] > 70 and trend_bearish:
            new_signal = -SIZE_ENTRY
        # Trend regime: Supertrend bearish + breakdown below Bollinger mid
        elif is_trend and st_bearish and below_mid and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Trend regime: 4h HMA bearish + Supertrend flip bearish
        elif trend_bearish and st_bearish and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Simple: RSI > 65 + Supertrend bearish (catches reversals)
        elif rsi[i] > 65 and st_bearish:
            new_signal = -SIZE_ENTRY
        # Simple: Price at Bollinger upper + 4h HMA bearish
        elif near_upper and trend_bearish:
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