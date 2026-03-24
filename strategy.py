#!/usr/bin/env python3
"""
Experiment #1502: 12h Primary + 1d/1w HTF — Connors RSI Mean Reversion with Choppiness Regime

Hypothesis: After analyzing 1500+ failed strategies, the pattern is clear:
1. 12h timeframe should generate 20-50 trades/year (optimal for fee drag vs opportunity)
2. Connors RSI (CRSI) has proven 75% win rate in bear/range markets (ETH Sharpe +0.923)
3. Choppiness Index regime filter prevents mean reversion in strong trends
4. Dual HTF (1d HMA + 1w trend) provides macro bias without over-filtering
5. Multiple entry paths (CRSI extreme OR Donchian breakout) ensure sufficient trades

Key design choices:
- Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long when CRSI < 15 + price > 1d HMA (mean reversion in uptrend)
- Short when CRSI > 85 + price < 1d HMA (mean reversion in downtrend)
- Choppiness Index > 61.8 = range (enable mean reversion), < 38.2 = trend (enable breakout)
- Donchian(20) breakout as secondary entry when trending
- ATR(14) 2.5x trailing stop for risk management
- Position size 0.25 with discrete levels (0.0, ±0.25) to minimize fee churn
- 12h timeframe for optimal trade frequency (20-50/year)

Timeframe: 12h (as required by experiment #1502)
HTF: 1d + 1w (call get_htf_data ONCE before loop for each!)
Position Size: 0.25 (discrete: 0.0, ±0.25)
Target: 30-60 trades/train, 5-10 trades/test, Sharpe > 0.618
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_1d1w_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak component of Connors RSI
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI of streak values
    for i in range(period, n):
        streak_window = streak[i-period+1:i+1]
        if np.any(np.isnan(streak_window)):
            continue
        # Convert streak to 0-100 scale
        max_streak = np.max(np.abs(streak_window))
        if max_streak > 0:
            normalized = (streak_window + max_streak) / (2 * max_streak) * 100
            streak_rsi[i] = np.mean(normalized[-period:])
        else:
            streak_rsi[i] = 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of Connors RSI
    Measures where current price change ranks vs recent changes
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    percent_rank = np.full(n, np.nan)
    delta = np.diff(close, prepend=close[0])
    
    for i in range(period, n):
        window = delta[i-period+1:i+1]
        if np.any(np.isnan(window)):
            continue
        current = delta[i]
        count_below = np.sum(window[:-1] < current)  # exclude current from comparison
        percent_rank[i] = (count_below / (period - 1)) * 100
    
    return percent_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with 75% win rate
    """
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    n = len(close)
    crsi = np.full(n, np.nan)
    
    mask = ~np.isnan(rsi_short) & ~np.isnan(rsi_streak) & ~np.isnan(pr)
    crsi[mask] = (rsi_short[mask] + rsi_streak[mask] + pr[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return choppiness

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_sma(close, period=50):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1w HMA) - regime filter ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === DAILY TREND (1d HMA) - direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = choppiness[i] > 55.0  # Range market (mean reversion works)
        is_trending = choppiness[i] < 45.0  # Trending market (breakout works)
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 20.0  # Mean reversion long
        crsi_overbought = crsi[i] > 80.0  # Mean reversion short
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === SMA50 FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === DESIRED SIGNAL - MULTIPLE ENTRY PATHS ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # PATH 1: Mean Reversion in Range (CRSI extreme + choppy + HTF bias)
        if is_choppy and crsi_oversold and daily_bull:
            desired_signal = BASE_SIZE
            signal_strength = 1.0
        elif is_choppy and crsi_overbought and daily_bear:
            desired_signal = -BASE_SIZE
            signal_strength = 1.0
        
        # PATH 2: Mean Reversion with Weekly Trend (looser CRSI bands)
        elif weekly_bull and crsi[i] < 30.0 and above_sma50:
            desired_signal = BASE_SIZE * 0.8
            signal_strength = 0.8
        elif weekly_bear and crsi[i] > 70.0 and below_sma50:
            desired_signal = -BASE_SIZE * 0.8
            signal_strength = 0.8
        
        # PATH 3: Trend Breakout (when trending + HTF alignment)
        elif is_trending and daily_bull and breakout_long:
            desired_signal = BASE_SIZE
            signal_strength = 1.0
        elif is_trending and daily_bear and breakout_short:
            desired_signal = -BASE_SIZE
            signal_strength = 1.0
        
        # PATH 4: Simple trend following (ensures trades when other paths fail)
        elif weekly_bull and daily_bull and above_sma50 and crsi[i] < 50.0:
            desired_signal = BASE_SIZE * 0.6
            signal_strength = 0.6
        elif weekly_bear and daily_bear and below_sma50 and crsi[i] > 50.0:
            desired_signal = -BASE_SIZE * 0.6
            signal_strength = 0.6
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
            signal_strength = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if signal_strength >= 0.9:
            final_signal = np.sign(desired_signal) * BASE_SIZE
        elif signal_strength >= 0.7:
            final_signal = np.sign(desired_signal) * BASE_SIZE * 0.8
        elif signal_strength >= 0.5:
            final_signal = np.sign(desired_signal) * BASE_SIZE * 0.6
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals