#!/usr/bin/env python3
"""
Experiment #341: 12h Regime-Adaptive Strategy with Choppiness Filter + Connors RSI + Fisher Transform
Hypothesis: Market regime detection (trend vs range) should improve entry quality.
Choppiness Index > 61.8 = range market (use mean-reversion with Connors RSI).
Choppiness Index < 38.2 = trending market (use breakout with Fisher confirmation).
12h timeframe captures multi-day swings, 1d HTF for macro bias, 1w for secular trend.
Key insight: Different logic per regime reduces whipsaw losses in 2022 crash and 2025 bear.
Target: Beat Sharpe=0.499 with regime-adaptive entries, 15-30 trades/year, DD < -30%.
Position sizing: 0.25 entry, 0.125 half-position at TP, stoploss at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_chop_crsi_fisher_daily_weekly_hma_atr_v1"
timeframe = "12h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    avg_sg = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_sl = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    rs_streak = np.where(avg_sl > 0, avg_sg / avg_sl, 100.0)
    rsi_streak = 100 - 100 / (1 + rs_streak)
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Percent Rank
    percent_rank = np.zeros(len(close))
    for i in range(rank_period, len(close)):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.66 * (price - LL) / (HH - LL) - 0.33
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    typical = (high + low) / 2.0
    highest = pd.Series(typical).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(typical).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    with np.errstate(divide='ignore', invalid='ignore'):
        x = 0.66 * (typical - lowest) / price_range - 0.33
    x = np.clip(x, -0.99, 0.99)  # Prevent division by zero in log
    
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    fisher = np.nan_to_num(fisher, nan=0.0)
    return fisher

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands for regime detection."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma * 100
    return upper, lower, bandwidth

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    fisher = calculate_fisher_transform(high, low, 9)
    chop = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    
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
    
    # Fisher transform signal tracking
    prev_fisher = 0.0
    
    for i in range(300, n):  # Start after 300 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            prev_fisher = fisher[i-1] if i > 0 else 0.0
            continue
        
        # === REGIME DETECTION ===
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        is_neutral = not is_trending and not is_ranging
        
        # === MACRO TREND BIAS ===
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        weekly_bullish = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        weekly_bearish = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = prev_fisher < -1.5 and fisher[i] >= -1.5
        fisher_cross_short = prev_fisher > 1.5 and fisher[i] <= 1.5
        prev_fisher = fisher[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # === BOLLINGER BAND SIGNALS ===
        bb_breakout_long = close[i] > bb_upper[i-1] if i > 0 else False
        bb_breakout_short = close[i] < bb_lower[i-1] if i > 0 else False
        bb_mean_revert_long = close[i] < bb_lower[i] and rsi[i] < 40
        bb_mean_revert_short = close[i] > bb_upper[i] and rsi[i] > 60
        
        new_signal = 0.0
        
        # === TRENDING REGIME (use breakout + momentum) ===
        if is_trending:
            # Long: Breakout + Daily bullish + Fisher confirmation
            if (bb_breakout_long or close[i] > bb_upper[i]) and daily_bullish and fisher[i] > -1.0:
                new_signal = SIZE_ENTRY
            # Short: Breakdown + Daily bearish + Fisher confirmation
            elif (bb_breakout_short or close[i] < bb_lower[i]) and daily_bearish and fisher[i] < 1.0:
                new_signal = -SIZE_ENTRY
            # Secondary: Fisher cross with weekly alignment
            elif fisher_cross_long and weekly_bullish and rsi[i] > 45:
                new_signal = SIZE_ENTRY
            elif fisher_cross_short and weekly_bearish and rsi[i] < 55:
                new_signal = -SIZE_ENTRY
        
        # === RANGING REGIME (use mean-reversion) ===
        elif is_ranging:
            # Long: CRSI oversold + price near BB lower + RSI low
            if crsi_oversold and bb_mean_revert_long:
                new_signal = SIZE_ENTRY
            # Short: CRSI overbought + price near BB upper + RSI high
            elif crsi_overbought and bb_mean_revert_short:
                new_signal = -SIZE_ENTRY
            # Secondary: Pure CRSI extreme with daily filter
            elif crsi[i] < 10 and daily_bullish:
                new_signal = SIZE_ENTRY
            elif crsi[i] > 90 and daily_bearish:
                new_signal = -SIZE_ENTRY
        
        # === NEUTRAL REGIME (mixed approach) ===
        else:
            # Use Fisher crosses with RSI filter
            if fisher_cross_long and rsi[i] > 40:
                new_signal = SIZE_ENTRY
            elif fisher_cross_short and rsi[i] < 60:
                new_signal = -SIZE_ENTRY
            # Or CRSI extremes
            elif crsi_oversold and rsi[i] < 35:
                new_signal = SIZE_ENTRY
            elif crsi_overbought and rsi[i] > 65:
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