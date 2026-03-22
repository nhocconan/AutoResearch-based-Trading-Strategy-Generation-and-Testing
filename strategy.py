#!/usr/bin/env python3
"""
Experiment #012: 1d Connors RSI Mean Reversion with 1w HMA Trend Bias
Hypothesis: Daily timeframe captures major swings with less noise. Connors RSI (CRSI)
combines RSI(3) + RSI_Streak(2) + PercentRank(100) for superior mean reversion signals.
1w HMA provides primary trend bias (HTF) to avoid counter-trend trades.
Choppiness Index detects regime: CHOP>61.8 = range (mean revert), CHOP<38.2 = trend.
Entry: CRSI<15 + price>1w_HMA (long), CRSI>85 + price<1w_HMA (short).
Exit: 2.5*ATR trailing stop or CRSI crosses 50.
Conservative sizing (0.25-0.30) with discrete levels to minimize fee churn.
Designed to work through 2022 crash and 2025 bear market with fewer but higher quality trades.
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_1w_hma_chop_regime_atr_v1"
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

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI Streak component of Connors RSI.
    Measures consecutive up/down days.
    """
    n = len(close)
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    for i in range(period, n):
        streak_window = streak[i-period+1:i+1]
        up_streak = np.sum(np.where(streak_window > 0, streak_window, 0))
        down_streak = np.abs(np.sum(np.where(streak_window < 0, streak_window, 0)))
        
        if down_streak > 0:
            rs = up_streak / down_streak
            streak_rsi[i] = 100 - 100 / (1 + rs)
        else:
            streak_rsi[i] = 100.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank component of Connors RSI.
    Measures current price change vs past period changes.
    """
    n = len(close)
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    
    for i in range(period, n):
        current_change = close[i] - close[i-1]
        past_changes = close[i-period+1:i] - close[i-period:i-1]
        count_below = np.sum(past_changes < current_change)
        pct_rank[i] = 100 * count_below / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_3 = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_3 + rsi_streak + pct_rank) / 3.0
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            atr_sum = np.sum(calculate_atr(high[i-period+1:i+1], low[i-period+1:i+1], close[i-period+1:i+1], 1))
            chop[i] = 100 * np.log10((highest_high - lowest_low) / atr_sum) / np.log10(period)
    
    return chop

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
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    
    # Additional trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):  # Need 250 bars for PR(100) + warmup
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # 1w trend bias (HTF)
        hma_1w_bullish = close[i] > hma_1w_aligned[i]
        hma_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and ema_50[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and ema_50[i] < ema_200[i]
        
        # Regime detection via Choppiness
        is_range = chop[i] > 55.0  # Slightly lower threshold for more trades
        is_trend = chop[i] < 45.0
        
        # CRSI extremes for mean reversion
        crsi_oversold = crsi[i] < 20.0  # Loosened from 15 for more trades
        crsi_overbought = crsi[i] > 80.0  # Loosened from 85 for more trades
        
        # CRSI cross 50 for exit
        crsi_cross_above_50 = crsi[i] > 50 and crsi[i-1] <= 50 if i > 0 else False
        crsi_cross_below_50 = crsi[i] < 50 and crsi[i-1] >= 50 if i > 0 else False
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # CRSI oversold + 1w HMA bullish (trend-aligned mean reversion)
        if crsi_oversold and hma_1w_bullish:
            new_signal = SIZE_ENTRY
        # CRSI very oversold (strong mean revert signal regardless of trend)
        elif crsi[i] < 10.0:
            new_signal = SIZE_ENTRY
        # Range market + CRSI oversold
        elif is_range and crsi_oversold and ema_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # CRSI overbought + 1w HMA bearish (trend-aligned mean reversion)
        if crsi_overbought and hma_1w_bearish:
            new_signal = -SIZE_ENTRY
        # CRSI very overbought (strong mean revert signal regardless of trend)
        elif crsi[i] > 90.0:
            new_signal = -SIZE_ENTRY
        # Range market + CRSI overbought
        elif is_range and crsi_overbought and ema_bearish:
            new_signal = -SIZE_ENTRY
        
        # === EXIT SIGNALS ===
        # CRSI crosses 50 (mean reversion complete)
        if position_side > 0 and crsi_cross_below_50:
            new_signal = 0.0
        if position_side < 0 and crsi_cross_above_50:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
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