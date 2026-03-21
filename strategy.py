#!/usr/bin/env python3
"""
Experiment #099: 1h Connors RSI + 4h HMA Trend + Choppiness Regime Filter
Hypothesis: Standard RSI fails in bear/range markets. Connors RSI (CRSI) has 75% win rate
for mean-reversion entries. Combine with 4h HMA trend filter (proven) and Choppiness Index
to adapt between trend-following (CHOP<38) and mean-reversion (CHOP>62). This should work
in both bull (2021) and bear/range (2022, 2025) markets. Position sizing: 0.25 entry,
0.15 half at 1.5R, stoploss 2.5*ATR. 1h timeframe gives 20-50 trades/year target.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h_hma_chop_regime_v1"
timeframe = "1h"
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
    Long entry: CRSI < 10 (oversold)
    Short entry: CRSI > 90 (overbought)
    """
    n = len(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - measures consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        up_streaks = np.sum(streak[max(0, i-streak_period+1):i+1] > 0)
        total = streak_period
        streak_rsi[i] = 100 * up_streaks / total if total > 0 else 50
    
    # Percent Rank (100) - percentile of price change over lookback
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        price_changes = np.diff(close[i-rank_period+1:i+1])
        current_change = close[i] - close[i-1] if i > 0 else 0
        if len(price_changes) > 0:
            percent_rank[i] = 100 * np.sum(price_changes < current_change) / len(price_changes)
        else:
            percent_rank[i] = 50
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean-reversion)
    CHOP < 38.2 = trending market (trend-following)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
        else:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                        abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_sma(close, period=200):
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
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
    
    for i in range(250, n):  # Need 250 for CRSI rank_period + SMA200
        # 4h trend filter (HTF)
        daily_bullish = close[i] > hma_4h_aligned[i]
        daily_bearish = close[i] < hma_4h_aligned[i]
        
        # SMA200 filter for long-term trend
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # Regime detection via Choppiness
        is_ranging = chop[i] > 55  # Mean-reversion regime
        is_trending = chop[i] < 45  # Trend-following regime
        
        # Connors RSI extremes
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # Connors RSI moderate extremes (for trending regime)
        crsi_moderate_oversold = crsi[i] < 30
        crsi_moderate_overbought = crsi[i] > 70
        
        new_signal = 0.0
        
        # LONG ENTRY - Mean Reversion Regime (ranging market)
        if is_ranging and crsi_oversold and above_sma200:
            new_signal = SIZE_ENTRY
        # LONG ENTRY - Trend Following Regime (trending market)
        elif is_trending and crsi_moderate_oversold and daily_bullish and above_sma200:
            new_signal = SIZE_ENTRY
        # LONG ENTRY - CRSI extreme oversold (any regime, strong signal)
        elif crsi[i] < 10 and above_sma200:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY - Mean Reversion Regime (ranging market)
        if is_ranging and crsi_overbought and below_sma200:
            new_signal = -SIZE_ENTRY
        # SHORT ENTRY - Trend Following Regime (trending market)
        elif is_trending and crsi_moderate_overbought and daily_bearish and below_sma200:
            new_signal = -SIZE_ENTRY
        # SHORT ENTRY - CRSI extreme overbought (any regime, strong signal)
        elif crsi[i] > 90 and below_sma200:
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
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 1.5 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if lowest_close == 0.0 or close[i] < lowest_close:
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
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
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