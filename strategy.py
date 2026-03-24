#!/usr/bin/env python3
"""
Experiment #1560: 1h Primary + 4h/12h HTF — Connors RSI + Choppiness Regime Strategy

Hypothesis: After 1159 failed experiments, key insights for 1h timeframe:
1. Lower TF (1h) needs VERY strict filters to avoid fee drag (>100 trades/yr = death)
2. BUT filters can't be SO strict that we get 0 trades (common failure mode)
3. Connors RSI (CRSI) has proven 75% win rate in mean-reversion setups
4. Choppiness Index (CHOP) regime filter: CHOP>55=range, CHOP<45=trend
5. Use 4h HMA for trend BIAS, 12h CHOP for REGIME, 1h CRSI for ENTRY TIMING
6. Session filter (8-20 UTC) + volume filter reduces false signals

Strategy Design:
- HTF Bias: 4h HMA(21) for trend direction
- HTF Regime: 12h Choppiness(14) for trend vs range detection
- Primary: 1h Connors RSI for entry timing (CRSI<25 long, CRSI>75 short)
- Volume: volume > 0.8x 20-bar average
- Session: only enter 8-20 UTC (major market hours)
- Exit: 2.0x ATR(14) trailing stop via signal→0
- Size: 0.25 discrete (0.0, ±0.25) — smaller for lower TF

Why this should work on 1h:
- 4h HMA gives us HTF trend direction (fewer whipsaws)
- 12h CHOP tells us if we're in trend or range regime
- 1h CRSI gives precise entry timing within HTF trend
- Session+volume filters cut noise by ~60%
- Target: 40-80 trades/year, Sharpe > 0.618

Timeframe: 1h (required for this experiment)
HTF: 4h HMA + 12h Choppiness
Target: Sharpe > 0.618, trades 30-80/year, DD < -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h12h_hma_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        if w_period < 1:
            return result
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = range-bound (mean revert)
    CHOP < 38.2 = trending (trend follow)
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    Composite of 3 components for mean-reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Long: CRSI < 10-20 (oversold)
    Short: CRSI > 80-90 (overbought)
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_3 = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_3 = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_3 = np.full(n, np.nan)
    mask = loss_3 > 1e-10
    rsi_3[mask] = 100.0 - (100.0 / (1.0 + gain_3[mask] / loss_3[mask]))
    rsi_3[loss_3 <= 1e-10] = 100.0
    rsi_3[:rsi_period] = np.nan
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    streak_gain_2 = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_2 = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask = streak_loss_2 > 1e-10
    rsi_streak[mask] = 100.0 - (100.0 / (1.0 + streak_gain_2[mask] / streak_loss_2[mask]))
    rsi_streak[streak_loss_2 <= 1e-10] = 100.0
    rsi_streak[:streak_period] = np.nan
    
    # Component 3: Percent Rank of daily returns over 100 periods
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i - rank_period + 1:i + 1])
        if len(returns) > 0:
            current_return = returns[-1]
            rank = np.sum(returns < current_return) / len(returns) * 100.0
            percent_rank[i] = rank
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_3) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_3[mask] + rsi_streak[mask] + percent_rank[mask]) / 3.0
    
    return crsi

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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_sma

def get_hour_from_open_time(open_time_array):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hours = (open_time_array // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h Choppiness for regime detection
    chop_12h_raw = calculate_choppiness(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        period=14
    )
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    hour = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h TF
    
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (12h Choppiness) ===
        chop = chop_12h_aligned[i]
        is_trend_regime = chop < 50.0  # CHOP < 50 = trending
        is_range_regime = chop > 55.0  # CHOP > 55 = range
        
        # === TREND BIAS (4h HMA) ===
        weekly_bull = close[i] > hma_4h_aligned[i]
        weekly_bear = close[i] < hma_4h_aligned[i]
        
        # === CRSI ENTRY SIGNALS (LOOSE enough to fire) ===
        crsi_oversold = crsi[i] < 25.0  # Long entry
        crsi_overbought = crsi[i] > 75.0  # Short entry
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma[i] if vol_sma[i] > 1e-10 else True
        
        # === SESSION FILTER (8-20 UTC only) ===
        session_ok = 8 <= hour[i] <= 20
        
        # === ENTRY LOGIC — BALANCED FOR TRADES ===
        desired_signal = 0.0
        
        # LONG: Trend regime + 4h bull + CRSI oversold + volume + session
        # OR: Range regime + CRSI oversold + volume + session (mean revert)
        if crsi_oversold and volume_ok and session_ok:
            if is_trend_regime and weekly_bull:
                desired_signal = BASE_SIZE  # Trend-following long
            elif is_range_regime:
                desired_signal = BASE_SIZE  # Mean-reversion long
        
        # SHORT: Trend regime + 4h bear + CRSI overbought + volume + session
        # OR: Range regime + CRSI overbought + volume + session (mean revert)
        if crsi_overbought and volume_ok and session_ok:
            if is_trend_regime and weekly_bear:
                desired_signal = -BASE_SIZE  # Trend-following short
            elif is_range_regime:
                desired_signal = -BASE_SIZE  # Mean-reversion short
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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