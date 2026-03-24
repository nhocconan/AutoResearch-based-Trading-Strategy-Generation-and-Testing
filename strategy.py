#!/usr/bin/env python3
"""
Experiment #1492: 12h Primary + 1d/1w HTF — Dual Regime Strategy with Choppiness Filter

Hypothesis: After 1114 failed strategies, the clearest pattern is:
1. Higher timeframes (12h, 1d) work better than lower TFs (30m, 1h, 4h)
2. Dual-regime strategies (trend in trending markets, mean-revert in choppy) show promise
3. Choppiness Index is the best regime filter (CHOP>61.8 = range, CHOP<38.2 = trend)
4. Connors RSI excels at mean reversion entries (75% win rate in research)
5. Donchian breakouts work best for trend following

This strategy uses:
- 1w HMA for ultra-macro trend bias (only trade with weekly trend)
- 1d HMA for macro trend direction
- 12h Choppiness Index for regime detection (trend vs mean-revert mode)
- REGIME 1 (CHOP<45): Donchian(20) breakout + RSI(14) momentum confirmation
- REGIME 2 (CHOP>55): Connors RSI mean reversion (CRSI<15 long, CRSI>85 short)
- ATR(14)*2.5 trailing stoploss on all positions
- Discrete signal sizes (0.0, ±0.25, ±0.30) to minimize fee churn

Why 12h + 1d/1w should work:
1. 12h = target 20-50 trades/year (fee drag ~1-2.5%)
2. Weekly HMA prevents trading against ultra-macro trend
3. Choppiness filter switches between trend/mean-revert modes
4. Connors RSI for mean reversion has proven 75% win rate
5. Loose entry filters ensure sufficient trades (≥10 train, ≥3 test)

Timeframe: 12h
HTF: 1d, 1w (call get_htf_data ONCE before loop!)
Position Size: 0.30 (discrete levels: 0.0, ±0.25, ±0.30)
Target: 20-50 trades/year, Sharpe > 0.618 (beat current best), ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_donchian_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        result = np.convolve(series, weights, mode='valid')
        return np.concatenate([np.full(window - 1, np.nan), result])
    
    close_series = pd.Series(close)
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    if len(wma_half) != len(wma_full):
        min_len = min(len(wma_half), len(wma_full))
        wma_half = wma_half[-min_len:]
        wma_full = wma_full[-min_len:]
    
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    # Pad to match original length
    pad_len = n - len(hma)
    if pad_len > 0:
        hma = np.concatenate([np.full(pad_len, np.nan), hma])
    
    return hma[:n]

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    choppiness = np.full(n, np.nan)
    
    for i in range(period * 2, n):
        if np.isnan(atr[i]):
            continue
        
        atr_sum = np.nansum(atr[i - period + 1:i + 1])
        highest_high = np.nanmax(high[i - period + 1:i + 1])
        lowest_low = np.nanmin(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - composite mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long: CRSI < 10-15, Short: CRSI > 85-90
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - RSI of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_values = streak[i - streak_period + 1:i + 1]
        gains = np.sum(np.maximum(streak_values, 0))
        losses = np.abs(np.sum(np.minimum(streak_values, 0)))
        if losses > 1e-10:
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + gains / losses))
        else:
            streak_rsi[i] = 100.0
    
    # Percent Rank - position of current return in last 100 returns
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close, prepend=close[0])
    for i in range(rank_period, n):
        window_returns = returns[i - rank_period + 1:i + 1]
        current_return = returns[i]
        count_below = np.sum(window_returns[:-1] < current_return)
        pct_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(pct_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + pct_rank[valid_mask]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for ultra-macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for macro trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start after all indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h[i]) or np.isnan(choppiness[i]):
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
        
        # === ULTRA-MACRO TREND (1w HMA) - only trade with weekly trend ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === MACRO TREND (1d HMA) - direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA) ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP < 38.2 = trending, CHOP > 61.8 = range/choppy
        # Use middle ground for smoother transitions
        is_trending = choppiness[i] < 45.0
        is_choppy = choppiness[i] > 55.0
        
        # === DONCHIAN BREAKOUT (for trending regime) ===
        breakout_high = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_low = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI MOMENTUM ===
        rsi_bullish = rsi[i] > 45.0
        rsi_bearish = rsi[i] < 55.0
        rsi_strong_bull = rsi[i] > 50.0
        rsi_strong_bear = rsi[i] < 50.0
        
        # === CONNORS RSI (for mean-reversion regime) ===
        crsi_oversold = not np.isnan(crsi[i]) and crsi[i] < 20.0
        crsi_overbought = not np.isnan(crsi[i]) and crsi[i] > 80.0
        
        # === DESIRED SIGNAL - DUAL REGIME ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND FOLLOWING MODE
            # LONG: Weekly bull + Daily bull + Donchian breakout + RSI support
            if weekly_bull and daily_bull and hma_bull:
                if breakout_high and rsi_bullish:
                    desired_signal = BASE_SIZE
                elif hma_bull and rsi_strong_bull and close[i] > hma_1d_aligned[i]:
                    desired_signal = BASE_SIZE * 0.8
            
            # SHORT: Weekly bear + Daily bear + Donchian breakout + RSI support
            elif weekly_bear and daily_bear and hma_bear:
                if breakout_low and rsi_bearish:
                    desired_signal = -BASE_SIZE
                elif hma_bear and rsi_strong_bear and close[i] < hma_1d_aligned[i]:
                    desired_signal = -BASE_SIZE * 0.8
        
        elif is_choppy:
            # MEAN REVERSION MODE (Connors RSI)
            # LONG: CRSI oversold + price above weekly HMA (don't fight macro)
            if crsi_oversold and weekly_bull:
                desired_signal = BASE_SIZE * 0.9
            
            # SHORT: CRSI overbought + price below weekly HMA
            elif crsi_overbought and weekly_bear:
                desired_signal = -BASE_SIZE * 0.9
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.2:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.2:
            final_signal = -BASE_SIZE * 0.5
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