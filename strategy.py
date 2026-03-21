#!/usr/bin/env python3
"""
Experiment #108: 1d Connors RSI + Choppiness Regime + Weekly HMA Trend
Hypothesis: Daily timeframe needs regime-adaptive logic. Use Choppiness Index
to detect range vs trend, then apply Connors RSI for mean reversion in ranges
and trend-following in trends. Weekly HMA provides HTF trend bias (proven).
Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 catches reversals
better than standard RSI. This should work in both bull (2021) and bear (2022, 2025).
Position sizing: 0.25 entry, 0.15 half at profit, stoploss 2.5*ATR trailing.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_weekly_hma_v1"
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
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
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
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Better at catching reversals than standard RSI.
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak component - measure consecutive up/down days
    streak_rsi = np.zeros(n)
    for i in range(1, n):
        streak = 0
        if close[i] > close[i-1]:
            j = i
            while j > 0 and close[j] > close[j-1]:
                streak += 1
                j -= 1
        elif close[i] < close[i-1]:
            j = i
            while j > 0 and close[j] < close[j-1]:
                streak -= 1
                j -= 1
        # Convert streak to RSI-like value (0-100)
        if streak > 0:
            streak_rsi[i] = min(100, streak * 50)
        elif streak < 0:
            streak_rsi[i] = max(0, 100 + streak * 50)
        else:
            streak_rsi[i] = 50
    
    # Smooth streak RSI
    streak_rsi_s = pd.Series(streak_rsi).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    # Percent Rank component - where does current price rank in last N days?
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine all three components
    for i in range(rank_period, n):
        crsi[i] = (rsi_short[i] + streak_rsi_s[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        atr_sum = np.sum(atr[i-period+1:i+1])
        
        if highest > lowest and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

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
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):  # Start after warmup period
        # Weekly trend filter (HTF) - price relative to Weekly HMA
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Long-term trend filter
        long_trend_bullish = close[i] > sma_200[i]
        long_trend_bearish = close[i] < sma_200[i]
        
        # Regime detection via Choppiness Index
        is_range = chop[i] > 55  # Range/choppy market
        is_trend = chop[i] < 45  # Trending market
        
        # Connors RSI signals
        crsi_oversold = crsi[i] < 15  # Extreme oversold
        crsi_overbought = crsi[i] > 85  # Extreme overbought
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        new_signal = 0.0
        
        # REGIME-ADAPTIVE LOGIC
        
        if is_range:
            # Mean reversion strategy in range markets
            # Long: CRSI oversold + weekly bullish bias OR long-term bullish
            if crsi_oversold and (weekly_bullish or long_trend_bullish):
                new_signal = SIZE_ENTRY
            # Short: CRSI overbought + weekly bearish bias OR long-term bearish
            elif crsi_overbought and (weekly_bearish or long_trend_bearish):
                new_signal = -SIZE_ENTRY
            # CRSI cross signals for faster entries
            elif crsi[i] < 20 and crsi_rising and weekly_bullish:
                new_signal = SIZE_ENTRY
            elif crsi[i] > 80 and crsi_falling and weekly_bearish:
                new_signal = -SIZE_ENTRY
        else:
            # Trend following strategy in trending markets
            # Long: Weekly bullish + CRSI not overbought + CRSI rising
            if weekly_bullish and long_trend_bullish and crsi[i] < 70 and crsi_rising:
                new_signal = SIZE_ENTRY
            # Short: Weekly bearish + CRSI not oversold + CRSI falling
            elif weekly_bearish and long_trend_bearish and crsi[i] > 30 and crsi_falling:
                new_signal = -SIZE_ENTRY
            # Pullback entries in trend
            elif weekly_bullish and crsi_oversold:
                new_signal = SIZE_ENTRY
            elif weekly_bearish and crsi_overbought:
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
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
                # Take profit at 1.5R
                profit = close[i] - entry_price
                risk = 2.5 * atr[i]
                if profit >= 1.5 * risk:
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
                # Take profit at 1.5R
                profit = entry_price - close[i]
                risk = 2.5 * atr[i]
                if profit >= 1.5 * risk:
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