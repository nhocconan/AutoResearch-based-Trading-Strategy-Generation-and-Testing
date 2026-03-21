#!/usr/bin/env python3
"""
Experiment #358: 4h Connors RSI + Choppiness Regime + Daily HMA Trend
Hypothesis: 4h timeframe with Connors RSI (CRSI) for mean reversion entries combined with
Choppiness Index regime detection adapts to both ranging and trending markets. Daily HMA
provides macro trend bias. CRSI<15 for longs, CRSI>85 for shorts with loose thresholds
to ensure sufficient trades. CHOP>61.8 = range (favor mean reversion), CHOP<38.2 = trend
(favor breakout). This adapts to 2022 crash (trend) and 2025 bear/range market.
Timeframe: 4h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 30-60 trades total, DD < -30%.
Key insight: CRSI has 75% win rate on reversals, CHOP filters false signals in wrong regimes.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_daily_hma_atr_v1"
timeframe = "4h"
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
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate on mean reversion entries.
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) - very short term
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI of streak - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak values
    streak_rsi = calculate_rsi(np.abs(streak), streak_period)
    # Adjust sign: positive streak = bullish RSI, negative = bearish
    streak_rsi_signed = np.where(streak >= 0, streak_rsi, 100 - streak_rsi)
    
    # PercentRank - where current price ranks in last N bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    for i in range(rank_period, n):
        crsi[i] = (rsi_short[i] + streak_rsi_signed[i] + percent_rank[i]) / 3.0
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    atr_vals = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr_vals[i-period+1:i+1])
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    
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
    
    for i in range(150, n):  # Start after 150 bars for CRSI rank_period
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # Daily macro trend bias (SOFT filter)
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # Regime detection via Choppiness
        is_ranging = chop[i] > 55.0  # Looser threshold for more signals
        is_trending = chop[i] < 45.0
        
        # CRSI extremes for mean reversion
        crsi_oversold = crsi[i] < 20.0  # Loose threshold for trades
        crsi_overbought = crsi[i] > 80.0
        
        # CRSI very extreme (stronger signal)
        crsi_extreme_long = crsi[i] < 12.0
        crsi_extreme_short = crsi[i] > 88.0
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: CRSI extreme oversold + ranging regime (mean reversion)
        if crsi_extreme_long and is_ranging:
            new_signal = SIZE_ENTRY
        # Secondary: CRSI oversold + daily bullish (trend pullback)
        elif crsi_oversold and daily_bullish:
            new_signal = SIZE_ENTRY
        # Tertiary: CRSI oversold alone (ensures trades in all regimes)
        elif crsi[i] < 15.0:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: CRSI extreme overbought + ranging regime (mean reversion)
        if crsi_extreme_short and is_ranging:
            new_signal = -SIZE_ENTRY
        # Secondary: CRSI overbought + daily bearish (trend pullback)
        elif crsi_overbought and daily_bearish:
            new_signal = -SIZE_ENTRY
        # Tertiary: CRSI overbought alone (ensures trades in all regimes)
        elif crsi[i] > 85.0:
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