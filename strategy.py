#!/usr/bin/env python3
"""
Experiment #444: 1d Connors RSI + Weekly HMA Bias + Choppiness Regime Filter
Hypothesis: Daily timeframe reduces fee drag and whipsaws. Connors RSI (CRSI) has
proven 75% win rate for mean reversion. Weekly HMA provides HTF trend bias.
Choppiness Index filters regime: CHOP>61.8 = range (use CRSI), CHOP<38.2 = trend
(use trend-follow). This adaptive approach should work in both bull and bear markets.
Position sizing 0.25-0.30, stoploss 2.5*ATR for daily timeframe. Multiple entry paths
ensure >=10 trades per symbol. Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_weekly_hma_chop_regime_atr_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # RSI on streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    delta_streak = np.diff(streak, prepend=streak[0])
    gain_s = np.where(delta_streak > 0, delta_streak, 0.0)
    loss_s = np.where(delta_streak < 0, -delta_streak, 0.0)
    avg_g_s = pd.Series(gain_s).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_l_s = pd.Series(loss_s).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    rs_s = np.where(avg_l_s > 0, avg_g_s / avg_l_s, 100.0)
    rsi_streak = 100 - 100 / (1 + rs_s)
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # PercentRank(100) - where current close ranks in last 100 days
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        rank = np.sum(window[:-1] < close[i]) / (rank_period - 1)
        if i >= rank_period:
            crsi[i] = (rsi_close[i] + rsi_streak[i] + rank * 100) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest > lowest:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    
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
    
    for i in range(250, n):  # Start after 250 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (HTF)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Regime detection via Choppiness Index
        is_ranging = chop[i] > 55.0  # Use 55 as threshold (between 38.2 and 61.8)
        is_trending = chop[i] < 45.0
        
        # SMA 200 filter
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # CRSI signals
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_very_oversold = crsi[i] < 10.0
        crsi_very_overbought = crsi[i] > 90.0
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: CRSI very oversold + Weekly bullish + Above SMA200 (strong mean reversion)
        if crsi_very_oversold and weekly_bullish and above_sma200:
            new_signal = SIZE_ENTRY
        # Path 2: CRSI oversold + Ranging market + Weekly bullish
        elif crsi_oversold and is_ranging and weekly_bullish:
            new_signal = SIZE_ENTRY
        # Path 3: CRSI oversold + Above SMA200 + ATR normal (not extreme volatility)
        elif crsi_oversold and above_sma200 and atr[i] < atr[i-20] * 2.0:
            new_signal = SIZE_ENTRY
        # Path 4: Weekly bullish + CRSI < 30 + Price pullback
        elif weekly_bullish and crsi[i] < 30 and close[i] < close[i-5]:
            new_signal = SIZE_ENTRY
        # Path 5: Trending market + Weekly bullish + CRSI < 40 (trend pullback)
        elif is_trending and weekly_bullish and crsi[i] < 40:
            new_signal = SIZE_ENTRY
        # Path 6: CRSI < 20 + Above SMA200 (simple mean reversion in uptrend)
        elif crsi[i] < 20 and above_sma200:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: CRSI very overbought + Weekly bearish + Below SMA200
        if crsi_very_overbought and weekly_bearish and below_sma200:
            new_signal = -SIZE_ENTRY
        # Path 2: CRSI overbought + Ranging market + Weekly bearish
        elif crsi_overbought and is_ranging and weekly_bearish:
            new_signal = -SIZE_ENTRY
        # Path 3: CRSI overbought + Below SMA200 + ATR normal
        elif crsi_overbought and below_sma200 and atr[i] < atr[i-20] * 2.0:
            new_signal = -SIZE_ENTRY
        # Path 4: Weekly bearish + CRSI > 70 + Price rally
        elif weekly_bearish and crsi[i] > 70 and close[i] > close[i-5]:
            new_signal = -SIZE_ENTRY
        # Path 5: Trending market + Weekly bearish + CRSI > 60 (trend pullback)
        elif is_trending and weekly_bearish and crsi[i] > 60:
            new_signal = -SIZE_ENTRY
        # Path 6: CRSI > 80 + Below SMA200 (simple mean reversion in downtrend)
        elif crsi[i] > 80 and below_sma200:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for daily timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for daily timeframe)
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