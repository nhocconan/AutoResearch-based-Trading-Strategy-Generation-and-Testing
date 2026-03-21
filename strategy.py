#!/usr/bin/env python3
"""
Experiment #168: 1d Multi-Signal Strategy with Weekly HMA Bias
Hypothesis: Daily timeframe needs multiple independent entry triggers (OR logic)
to generate sufficient trades. Combining Connors RSI mean reversion, MACD momentum,
and simple trend pullbacks with loose thresholds. Weekly HMA provides macro bias
but doesn't block all entries. Key insight from failures: too many AND conditions
= 0 trades. Use OR logic so any trigger can enter. Position sizing 0.30 with
2.5*ATR stoploss. Targets both 2021 bull (trend) and 2022/2025 bear (mean revert).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_multi_signal_weekly_hma_ors_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    Reference: Connors et al.
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_abs = np.abs(streak)
    streak_delta = np.diff(streak_abs, prepend=streak_abs[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    avg_sg = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_sl = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    rs_streak = np.where(avg_sl > 0, avg_sg / avg_sl, 100.0)
    rsi_streak = 100 - 100 / (1 + rs_streak)
    
    # Percent Rank - where does current close rank vs last 100 days
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window[:-1] < window[-1]) / (rank_period - 1) * 100
        crsi[i] = (rsi_short[i] + rsi_streak[i] + rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    sma_200 = calculate_sma(close, 200)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
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
    
    for i in range(200, n):
        # Weekly trend bias (not a hard filter, just influences direction)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Daily trend
        daily_bullish = hma_21[i] > hma_50[i]
        daily_bearish = hma_21[i] < hma_50[i]
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # RSI signals (loose thresholds for more trades)
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_rising = rsi_14[i] > rsi_14[i-2] if i > 2 else False
        rsi_falling = rsi_14[i] < rsi_14[i-2] if i > 2 else False
        
        # CRSI extreme signals (mean reversion)
        crsi_oversold = crsi[i] < 20  # Loosened from 10
        crsi_overbought = crsi[i] > 80  # Loosened from 90
        
        # MACD signals
        macd_bullish_cross = macd_hist[i] > 0 and macd_hist[i-1] <= 0 if i > 0 else False
        macd_bearish_cross = macd_hist[i] < 0 and macd_hist[i-1] >= 0 if i > 0 else False
        macd_positive = macd_hist[i] > 0
        macd_negative = macd_hist[i] < 0
        
        # HMA crossover
        hma_cross_bull = hma_21[i] > hma_50[i] and hma_21[i-1] <= hma_50[i-1]
        hma_cross_bear = hma_21[i] < hma_50[i] and hma_21[i-1] >= hma_50[i-1]
        
        new_signal = 0.0
        
        # === SIGNAL 1: CRSI Mean Reversion (high probability, works in any regime) ===
        if crsi_oversold and rsi_rising:
            # Long on extreme oversold + RSI turning up
            new_signal = SIZE_ENTRY
        
        elif crsi_overbought and rsi_falling:
            # Short on extreme overbought + RSI turning down
            new_signal = -SIZE_ENTRY
        
        # === SIGNAL 2: RSI + SMA200 Filter (classic mean reversion) ===
        elif rsi_oversold and above_sma200:
            # Long: RSI oversold but still above long-term trend
            new_signal = SIZE_ENTRY
        
        elif rsi_overbought and below_sma200:
            # Short: RSI overbought but below long-term trend
            new_signal = -SIZE_ENTRY
        
        # === SIGNAL 3: MACD Momentum (trend following) ===
        elif macd_bullish_cross and (weekly_bullish or daily_bullish):
            # Long on MACD cross + trend confirmation
            new_signal = SIZE_ENTRY
        
        elif macd_bearish_cross and (weekly_bearish or daily_bearish):
            # Short on MACD cross + trend confirmation
            new_signal = -SIZE_ENTRY
        
        # === SIGNAL 4: HMA Crossover (simple trend) ===
        elif hma_cross_bull and macd_positive:
            # Long on HMA cross + MACD positive
            new_signal = SIZE_ENTRY
        
        elif hma_cross_bear and macd_negative:
            # Short on HMA cross + MACD negative
            new_signal = -SIZE_ENTRY
        
        # === SIGNAL 5: Pullback in Trend (buy dips in uptrend) ===
        elif daily_bullish and rsi_14[i] < 45 and rsi_rising:
            # Long pullback in uptrend
            new_signal = SIZE_ENTRY
        
        elif daily_bearish and rsi_14[i] > 55 and rsi_falling:
            # Short rally in downtrend
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