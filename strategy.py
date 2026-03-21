#!/usr/bin/env python3
"""
Experiment #057: 1h Regime-Adaptive CRSI + Fisher + 4h HMA Trend Filter
Hypothesis: Combine Connors RSI (CRSI) for mean reversion in ranging markets
with Fisher Transform for trend reversals. Use Choppiness Index to detect
market regime and switch logic accordingly. 4h HMA provides trend bias filter.
CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
Long: CRSI<10 in range market OR Fisher<-1.5 in trend market + 4h HMA bullish
Short: CRSI>90 in range market OR Fisher>+1.5 in trend market + 4h HMA bearish
This should work in both 2021-2024 bull and 2025 bear/range markets.
Position sizing: 0.25 entry, stoploss at 2.5*ATR, reduce to 0.12 at 2R profit.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_fisher_4h_hma_v1"
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
    """
    n = len(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        pos_streaks = np.sum(streak_vals > 0)
        if streak_period > 0:
            streak_rsi[i] = 100 * pos_streaks / streak_period
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100)
    pct_rank = np.zeros(n)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            pct_rank[i] = 100 * np.sum(returns[:-1] < current_return) / (len(returns) - 1)
        else:
            pct_rank[i] = 50
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + pct_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    where X = 0.66 * ((close - lowest) / (highest - lowest) - 0.5)
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        if highest > lowest:
            x = 0.66 * ((close[i] - lowest) / (highest - lowest) - 0.5)
            x = np.clip(x, -0.99, 0.99)  # Prevent division by zero
            fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0
        
        trigger[i] = fisher[i-1] if i > 0 else 0
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * (SUM(ATR, n) / (Highest High - Lowest Low)) / (ln(n))
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest > lowest and atr_sum > 0:
            chop[i] = 100 * (atr_sum / (highest - lowest)) / np.log(period)
        else:
            chop[i] = chop[i-1] if i > 0 else 50
    
    return chop

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
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
    rsi = calculate_rsi(close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    fisher, fisher_trigger = calculate_fisher(close, 9)
    chop = calculate_choppiness(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # 1h HMA for local trend
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    
    for i in range(100, n):
        # 4h trend filter (HTF)
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # Market regime detection
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # CRSI extreme levels for mean reversion
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # Fisher Transform signals
        fisher_long = fisher[i] < -1.5 and fisher_trigger[i] >= -1.5
        fisher_short = fisher[i] > 1.5 and fisher_trigger[i] <= 1.5
        
        # Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # 1h HMA trend
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        new_signal = 0.0
        
        # LONG ENTRY Logic
        if is_ranging:
            # Mean reversion in range market
            if crsi_oversold and trend_bullish:
                new_signal = SIZE_ENTRY
            elif crsi_oversold and hma_trend_long:
                new_signal = SIZE_ENTRY
        elif is_trending:
            # Trend following in trending market
            if fisher_long and trend_bullish:
                new_signal = SIZE_ENTRY
            elif st_long and trend_bullish and hma_trend_long:
                new_signal = SIZE_ENTRY
        else:
            # Neutral regime - use conservative signals
            if crsi_oversold and trend_bullish and st_long:
                new_signal = SIZE_ENTRY
            elif fisher_long and trend_bullish:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY Logic
        if is_ranging:
            # Mean reversion in range market
            if crsi_overbought and trend_bearish:
                new_signal = -SIZE_ENTRY
            elif crsi_overbought and hma_trend_short:
                new_signal = -SIZE_ENTRY
        elif is_trending:
            # Trend following in trending market
            if fisher_short and trend_bearish:
                new_signal = -SIZE_ENTRY
            elif st_short and trend_bearish and hma_trend_short:
                new_signal = -SIZE_ENTRY
        else:
            # Neutral regime - use conservative signals
            if crsi_overbought and trend_bearish and st_short:
                new_signal = -SIZE_ENTRY
            elif fisher_short and trend_bearish:
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = close[i] - entry_price
                    risk = 2.5 * atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = SIZE_HALF
                        position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = entry_price - close[i]
                    risk = 2.5 * atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = -SIZE_HALF
                        position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals