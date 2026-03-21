#!/usr/bin/env python3
"""
Experiment #362: 30m Bollinger Mean Reversion + 4h HMA Trend Filter + RSI Confirmation + ATR Stop
Hypothesis: 30m timeframe is ideal for mean reversion during range markets (2022, 2025 bear).
Bollinger Band squeezes indicate low vol → mean reversion likely. 4h HMA provides trend bias.
RSI(7) extremes (not 14) catch short-term oversold/overbought for faster entries.
ATR(14) stoploss at 2.0x protects during trend breaks. Build on #359 (Sharpe=0.061) by adding regime filter.
Timeframe: 30m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 40-80 trades total across train+test.
Key insight: 30m mean reversion works in bear/range markets where trend following fails.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_bollinger_mr_4h_hma_rsi7_atr_v1"
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

def calculate_rsi(close, period=7):
    """Calculate RSI indicator with shorter period for 30m."""
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
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return upper, lower, sma, bandwidth

def calculate_percentile_rank(close, period=100):
    """Calculate percentile rank for Connors RSI component."""
    n = len(close)
    pr = np.zeros(n)
    for i in range(period, n):
        window = close[i - period + 1:i + 1]
        current = close[i]
        pr[i] = np.sum(window < current) / period * 100
    return pr

def calculate_streak_rsi(close, period=2):
    """Calculate streak-based RSI for Connors RSI."""
    n = len(close)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = streak[i - 1]
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    for i in range(period, n):
        pos_streaks = np.sum(streak[i - period + 1:i + 1] > 0)
        streak_rsi[i] = pos_streaks / period * 100
    return streak_rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Calculate Connors RSI."""
    rsi = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    pr = calculate_percentile_rank(close, pr_period)
    crsi = (rsi + streak_rsi + pr) / 3
    return crsi

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
    rsi = calculate_rsi(close, 7)  # Faster RSI for 30m
    bb_upper, bb_lower, bb_sma, bb_bw = calculate_bollinger(close, 20, 2.0)
    
    # Calculate Connors RSI for mean reversion signals
    crsi = calculate_crsi(close, 3, 2, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (SOFT filter - doesn't block trades, just biases)
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # Bollinger Band position
        price_vs_upper = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if bb_upper[i] != bb_lower[i] else 0.5
        near_lower = price_vs_upper < 0.15  # Price in bottom 15% of BB
        near_upper = price_vs_upper > 0.85  # Price in top 15% of BB
        
        # Bollinger Band squeeze (low volatility = mean reversion likely)
        bb_squeeze = bb_bw[i] < np.nanpercentile(bb_bw[:i], 30) if i > 100 else False
        
        # RSI extremes for 30m mean reversion
        rsi_oversold = rsi[i] < 25
        rsi_overbought = rsi[i] > 75
        
        # Connors RSI extremes (more sensitive)
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # Volume confirmation (optional boost)
        vol_ma = pd.Series(prices["volume"].values).rolling(20, min_periods=20).mean().values
        high_volume = prices["volume"].values[i] > vol_ma[i] * 1.5 if not np.isnan(vol_ma[i]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (Mean Reversion) ===
        # Primary: Price near BB lower + RSI oversold + CRSI oversold
        if near_lower and rsi_oversold and crsi_oversold:
            new_signal = SIZE_ENTRY
        # Secondary: Price near BB lower + RSI oversold (CRSI neutral ok)
        elif near_lower and rsi_oversold and rsi[i] < 30:
            new_signal = SIZE_ENTRY
        # Tertiary: CRSI very oversold + trend bullish (trend pullback)
        elif crsi_oversold and trend_bullish and rsi[i] < 35:
            new_signal = SIZE_ENTRY
        # Quaternary: BB squeeze breakout long + trend bullish
        elif bb_squeeze and close[i] > bb_sma[i] and trend_bullish and rsi[i] > 40 and rsi[i] < 60:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (Mean Reversion) ===
        # Primary: Price near BB upper + RSI overbought + CRSI overbought
        if near_upper and rsi_overbought and crsi_overbought:
            new_signal = -SIZE_ENTRY
        # Secondary: Price near BB upper + RSI overbought (CRSI neutral ok)
        elif near_upper and rsi_overbought and rsi[i] > 70:
            new_signal = -SIZE_ENTRY
        # Tertiary: CRSI very overbought + trend bearish (trend pullback)
        elif crsi_overbought and trend_bearish and rsi[i] > 65:
            new_signal = -SIZE_ENTRY
        # Quaternary: BB squeeze breakout short + trend bearish
        elif bb_squeeze and close[i] < bb_sma[i] and trend_bearish and rsi[i] > 40 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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