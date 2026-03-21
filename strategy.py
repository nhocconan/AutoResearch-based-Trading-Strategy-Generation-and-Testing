#!/usr/bin/env python3
"""
Experiment #258: 1d Connors RSI + Choppiness Regime + Weekly HMA Trend Filter
Hypothesis: Daily timeframe with Connors RSI (CRSI) for mean reversion entries combined with 
Choppiness Index regime detection can adapt to both trending and ranging markets. Weekly HMA 
provides macro trend bias. CRSI<20 for long, CRSI>80 for short (looser than typical 10/90 to 
ensure trades). CHOP>61.8 = range (favor mean reversion), CHOP<38.2 = trend (favor trend follow).
This differs from failed experiments by using CRSI instead of simple RSI, and adding regime 
adaptation. Position sizing: 0.25 entry, 0.12 half at 2R. Stoploss: 2.5*ATR trailing.
Target: Beat Sharpe=0.499 with fewer but higher quality trades on 1d timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_weekly_hma_atr_v1"
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
    Streak: consecutive days up/down (positive/negative count)
    PercentRank: percentage of prior closes lower than current close
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
    
    # Convert streak to RSI-like value (0-100)
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    avg_streak_g = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_l = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_rs = np.where(avg_streak_l > 0, avg_streak_g / avg_streak_l, 100.0)
    streak_rsi = 100 - 100 / (1 + streak_rs)
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # PercentRank component (percentage of prior closes lower than current)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_lower = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_lower / rank_period
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range > 0, price_range, 1e-10)
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

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
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # Track previous values
    prev_crsi = np.roll(crsi, 1)
    prev_crsi[0] = crsi[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):  # Need 250 bars for all indicators to warm up
        # Weekly trend filter
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Long-term trend filter
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # Regime detection
        ranging_market = chop[i] > 55.0  # Slightly lower threshold to get more signals
        trending_market = chop[i] < 45.0
        
        # CRSI mean reversion signals (looser thresholds for more trades)
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_rising = crsi[i] > prev_crsi[i]
        crsi_falling = crsi[i] < prev_crsi[i]
        
        # CRSI cross signals
        crsi_cross_up = prev_crsi[i] <= 25.0 and crsi[i] > 25.0
        crsi_cross_down = prev_crsi[i] >= 75.0 and crsi[i] < 75.0
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Mean reversion in ranging market
        if ranging_market and crsi_oversold:
            if weekly_bullish or above_sma200:
                new_signal = SIZE_ENTRY
            elif crsi_rising:
                new_signal = SIZE_ENTRY
        
        # CRSI cross up with trend confirmation
        if crsi_cross_up:
            if weekly_bullish and above_sma200:
                new_signal = SIZE_ENTRY
            elif ranging_market and crsi_rising:
                new_signal = SIZE_ENTRY
        
        # Trending market pullback
        if trending_market and weekly_bullish:
            if crsi[i] < 40.0 and crsi_rising:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Mean reversion in ranging market
        if ranging_market and crsi_overbought:
            if weekly_bearish or below_sma200:
                new_signal = -SIZE_ENTRY
            elif crsi_falling:
                new_signal = -SIZE_ENTRY
        
        # CRSI cross down with trend confirmation
        if crsi_cross_down:
            if weekly_bearish and below_sma200:
                new_signal = -SIZE_ENTRY
            elif ranging_market and crsi_falling:
                new_signal = -SIZE_ENTRY
        
        # Trending market pullback
        if trending_market and weekly_bearish:
            if crsi[i] > 60.0 and crsi_falling:
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