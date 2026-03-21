#!/usr/bin/env python3
"""
Experiment #086: 30m Connors RSI + 4h/1d HMA Trend + Choppiness Regime Filter
Hypothesis: 30m timeframe needs faster entries than 12h/4h strategies. Connors RSI
(CRSI) has proven 75% win rate for mean reversion. Combine with 4h HMA trend filter
(proven in best strategies) and 1d HMA for higher-level regime. Add Choppiness Index
to switch between mean-reversion (CHOP>61.8) and trend-following (CHOP<38.2) modes.
This should work in both 2021-2022 trending and 2025 bear/range markets.
Position sizing: 0.25 entry, 0.125 at 1.5R profit, stoploss at 2.5*ATR trailing.
Timeframe: 30m (required for this experiment), HTF: 4h + 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_4h_1d_hma_chop_regime_v1"
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
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_g = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_l = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.fillna(50).values
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # RSI Streak - RSI of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = (-streak_delta).where(streak_delta < 0, 0.0)
    avg_streak_g = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_l = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_g / avg_streak_l.replace(0, np.nan)
    rsi_streak = 100 - 100 / (1 + streak_rs)
    rsi_streak = rsi_streak.fillna(50).values
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Percent Rank - percentage of closes lower than current in lookback
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        lookback = close[i-rank_period:i]
        count_lower = np.sum(lookback < close[i])
        percent_rank[i] = (count_lower / rank_period) * 100
    
    # CRSI calculation
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over period
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - low_s.rolling(window=period, min_periods=period).min()
    
    # CHOP formula
    chop = 100 * np.log10(tr_sum / hh_ll) / np.log10(period)
    chop = chop.fillna(50).values
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(200, n):
        # HTF Trend filters
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        hma_1d_bullish = close[i] > hma_1d_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # SMA 200 filter
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # Choppiness regime
        chop_range = chop[i] > 61.8  # Range market - mean revert
        chop_trend = chop[i] < 38.2  # Trending market - trend follow
        chop_neutral = not chop_range and not chop_trend
        
        # CRSI signals
        crsi_oversold = crsi[i] < 15  # Extreme oversold
        crsi_overbought = crsi[i] > 85  # Extreme overbought
        crsi_recover_long = crsi[i] > 20 and crsi[i-1] <= 20  # CRSI crossing above 20
        crsi_recover_short = crsi[i] < 80 and crsi[i-1] >= 80  # CRSI crossing below 80
        
        new_signal = 0.0
        
        # LONG ENTRY - Mean Reversion Mode (Range Market)
        if chop_range:
            # CRSI oversold + above 4h HMA + above SMA200
            if crsi_oversold and hma_4h_bullish and above_sma200:
                new_signal = SIZE_ENTRY
            # CRSI recovery + 4h HMA bullish
            elif crsi_recover_long and hma_4h_bullish:
                new_signal = SIZE_ENTRY
        
        # LONG ENTRY - Trend Mode (Trending Market)
        elif chop_trend:
            # Pullback to 4h HMA + CRSI not overbought
            if hma_4h_bullish and hma_1d_bullish and crsi[i] < 70:
                # Price pulled back near 4h HMA
                pullback_dist = (close[i] - hma_4h_aligned[i]) / hma_4h_aligned[i]
                if pullback_dist < 0.02 and pullback_dist > -0.05:  # Within 2% above or 5% below HMA
                    new_signal = SIZE_ENTRY
        
        # LONG ENTRY - Neutral Mode
        elif chop_neutral:
            if crsi_oversold and hma_4h_bullish and above_sma200:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY - Mean Reversion Mode (Range Market)
        if chop_range:
            # CRSI overbought + below 4h HMA + below SMA200
            if crsi_overbought and hma_4h_bearish and below_sma200:
                new_signal = -SIZE_ENTRY
            # CRSI recovery down + 4h HMA bearish
            elif crsi_recover_short and hma_4h_bearish:
                new_signal = -SIZE_ENTRY
        
        # SHORT ENTRY - Trend Mode (Trending Market)
        elif chop_trend:
            # Rally to 4h HMA + CRSI not oversold
            if hma_4h_bearish and hma_1d_bearish and crsi[i] > 30:
                # Price rallied near 4h HMA
                rally_dist = (close[i] - hma_4h_aligned[i]) / hma_4h_aligned[i]
                if rally_dist > -0.02 and rally_dist < 0.05:  # Within 2% below or 5% above HMA
                    new_signal = -SIZE_ENTRY
        
        # SHORT ENTRY - Neutral Mode
        elif chop_neutral:
            if crsi_overbought and hma_4h_bearish and below_sma200:
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