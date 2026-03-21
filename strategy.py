#!/usr/bin/env python3
"""
Experiment #123: 1h Hybrid Trend/Mean-Reversion with 4h HMA + CRSI + Choppiness
Hypothesis: Pure trend following failed in 2022 crash and 2025 bear market.
A hybrid approach that adapts to regime (trend vs range) using Choppiness Index
will outperform. Long when: 4h HMA bullish + CRSI<20 (oversold pullback) in trend,
OR CRSI<10 (extreme oversold) in range. Short when: 4h HMA bearish + CRSI>80
in trend, OR CRSI>90 in range. This catches both trend continuations and
mean-reversion opportunities. Position sizing: 0.25 entry, stoploss 2*ATR.
1h timeframe provides good trade frequency (50-100 trades/year target).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hybrid_crsi_chop_4h_hma_v1"
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

def calculate_rsi(close, period):
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
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    Streak RSI: RSI of consecutive up/down days
    PercentRank: percentage of prior returns lower than current
    """
    n = len(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to positive values for RSI calculation
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    # RSI on streak (simplified - treat streak magnitude as "gain/loss")
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        gains = 0
        losses = 0
        for j in range(i - streak_period + 1, i + 1):
            if streak[j] > 0:
                gains += streak_abs[j]
            elif streak[j] < 0:
                losses += streak_abs[j]
        if losses == 0:
            streak_rsi[i] = 100
        else:
            rs = gains / losses
            streak_rsi[i] = 100 - 100 / (1 + rs)
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # PercentRank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        returns = np.diff(close[max(0, i-rank_period):i+1])
        if len(returns) > 0 and len(returns) > 0:
            current_return = close[i] - close[i-1] if i > 0 else 0
            count_lower = np.sum(returns < current_return)
            percent_rank[i] = 100 * count_lower / len(returns)
    
    # Combine components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
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
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):  # Start after warmup period
        # 4h trend filter (HTF)
        daily_bullish = close[i] > hma_4h_aligned[i] if not np.isnan(hma_4h_aligned[i]) else False
        daily_bearish = close[i] < hma_4h_aligned[i] if not np.isnan(hma_4h_aligned[i]) else False
        
        # Regime detection via Choppiness Index
        is_trending = chop[i] < 50  # Below 50 = trending
        is_ranging = chop[i] >= 50  # Above 50 = ranging/choppy
        
        # CRSI extremes for mean reversion
        crsi_oversold = crsi[i] < 20
        crsi_extreme_oversold = crsi[i] < 10
        crsi_overbought = crsi[i] > 80
        crsi_extreme_overbought = crsi[i] > 90
        
        # Price above/below SMA200 for additional trend filter
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        price_below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        new_signal = 0.0
        
        # LONG ENTRY conditions (hybrid: trend + mean reversion)
        # Condition 1: Trending + 4h bullish + CRSI oversold pullback
        if is_trending and daily_bullish and crsi_oversold and price_above_sma200:
            new_signal = SIZE_ENTRY
        # Condition 2: Ranging + CRSI extreme oversold (pure mean reversion)
        elif is_ranging and crsi_extreme_oversold:
            new_signal = SIZE_ENTRY
        # Condition 3: 4h bullish + CRSI very low (strong trend pullback)
        elif daily_bullish and crsi[i] < 15 and price_above_sma200:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Condition 1: Trending + 4h bearish + CRSI overbought pullback
        if is_trending and daily_bearish and crsi_overbought and price_below_sma200:
            new_signal = -SIZE_ENTRY
        # Condition 2: Ranging + CRSI extreme overbought (pure mean reversion)
        elif is_ranging and crsi_extreme_overbought:
            new_signal = -SIZE_ENTRY
        # Condition 3: 4h bearish + CRSI very high (strong trend pullback)
        elif daily_bearish and crsi[i] > 85 and price_below_sma200:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = SIZE_EXIT
            
            # Take profit at 2R (reduce position)
            elif position_side > 0:
                profit = close[i] - entry_price
                risk = 2.0 * atr[i]
                if profit >= 2.0 * risk and np.abs(signals[i-1]) > 0:
                    new_signal = SIZE_EXIT  # Close position at target
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            # Calculate trailing stop (2*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = SIZE_EXIT
            
            # Take profit at 2R (reduce position)
            elif position_side < 0:
                profit = entry_price - close[i]
                risk = 2.0 * atr[i]
                if profit >= 2.0 * risk and np.abs(signals[i-1]) > 0:
                    new_signal = SIZE_EXIT  # Close position at target
        
        # CRSI exit signals (mean reversion complete)
        if position_side > 0 and crsi[i] > 70:
            new_signal = SIZE_EXIT  # Exit long when CRSI overbought
        if position_side < 0 and crsi[i] < 30:
            new_signal = SIZE_EXIT  # Exit short when CRSI oversold
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals