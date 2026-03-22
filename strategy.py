#!/usr/bin/env python3
"""
Experiment #474: 1d CRSI Mean Reversion + Weekly Trend Bias + ATR Stop
Hypothesis: Daily timeframe with Connors RSI (CRSI) for mean reversion entries
combined with weekly HMA trend filter. CRSI catches oversold/overbought extremes
(75% win rate in literature) while weekly bias prevents counter-trend trades.
1d timeframe = fewer trades but higher quality, less fee drag. 2.5*ATR stoploss.
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_weekly_hma_mean_reversion_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
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

def calculate_streak_rsi(close, period=2):
    """Calculate RSI of up/down streak lengths (Connors RSI component)."""
    n = len(close)
    streak = np.zeros(n)
    streak_direction = np.zeros(n)  # 1 = up streak, -1 = down streak
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streak_direction[i-1] >= 0:
                streak[i] = streak[i-1] + 1
                streak_direction[i] = 1
            else:
                streak[i] = 1
                streak_direction[i] = 1
        elif close[i] < close[i-1]:
            if streak_direction[i-1] <= 0:
                streak[i] = streak[i-1] + 1
                streak_direction[i] = -1
            else:
                streak[i] = 1
                streak_direction[i] = -1
        else:
            streak[i] = streak[i-1]
            streak_direction[i] = streak_direction[i-1]
    
    # Calculate RSI of streak values
    streak_rsi = calculate_rsi(streak, period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank of current close vs last period closes (Connors RSI component)."""
    n = len(close)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        pr[i] = (count_below / (period - 1)) * 100
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Calculate Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    hma_1d = calculate_hma(close, 21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (HTF)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend filter
        daily_bullish = close[i] > hma_1d[i]
        daily_bearish = close[i] < hma_1d[i]
        
        # CRSI mean reversion signals (Connors RSI extremes)
        crsi_oversold = crsi[i] < 15  # Strong buy signal
        crsi_overbought = crsi[i] > 85  # Strong sell signal
        crsi_moderate_oversold = crsi[i] < 25  # Moderate buy
        crsi_moderate_overbought = crsi[i] > 75  # Moderate sell
        
        # Additional filter: price not too far from HMA (avoid catching falling knife)
        price_vs_hma_ratio = close[i] / hma_1d[i]
        not_too_extended_long = price_vs_hma_ratio > 0.85  # Not down >15%
        not_too_extended_short = price_vs_hma_ratio < 1.20  # Not up >20%
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Weekly bullish + CRSI oversold + Daily bullish
        if weekly_bullish and crsi_oversold and daily_bullish:
            new_signal = SIZE_ENTRY
        # Path 2: Weekly bullish + CRSI moderate oversold + Daily bullish + not extended
        elif weekly_bullish and crsi_moderate_oversold and daily_bullish and not_too_extended_long:
            new_signal = SIZE_ENTRY
        # Path 3: Weekly bullish + CRSI < 30 + price above weekly HMA
        elif weekly_bullish and crsi[i] < 30 and close[i] > hma_1w_aligned[i]:
            new_signal = SIZE_ENTRY
        # Path 4: CRSI very oversold (<10) regardless of trend (strong MR signal)
        elif crsi[i] < 10 and not_too_extended_long:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Weekly bearish + CRSI overbought + Daily bearish
        if weekly_bearish and crsi_overbought and daily_bearish:
            new_signal = -SIZE_ENTRY
        # Path 2: Weekly bearish + CRSI moderate overbought + Daily bearish + not extended
        elif weekly_bearish and crsi_moderate_overbought and daily_bearish and not_too_extended_short:
            new_signal = -SIZE_ENTRY
        # Path 3: Weekly bearish + CRSI > 70 + price below weekly HMA
        elif weekly_bearish and crsi[i] > 70 and close[i] < hma_1w_aligned[i]:
            new_signal = -SIZE_ENTRY
        # Path 4: CRSI very overbought (>90) regardless of trend (strong MR signal)
        elif crsi[i] > 90 and not_too_extended_short:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 1d timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 1d timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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