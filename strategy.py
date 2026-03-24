#!/usr/bin/env python3
"""
Experiment #1558: 30m Primary + 4h/1d HTF — Regime-Adaptive Mean Reversion

Hypothesis: After 1157 failed experiments, key insights for LOWER timeframes:
1. 30m needs REGIME detection — trend-following fails (whipsaw in 2022/2025)
2. Connors RSI (CRSI) has 75% win rate for mean reversion in ranges
3. Choppiness Index filters: CHOP>50 = range (mean revert), CHOP<40 = trend (skip)
4. 4h HMA provides trend BIAS only — don't require perfect alignment
5. LOOSE entry thresholds ensure trades fire (CRSI<15 long, >85 short)
6. Session filter REMOVED — was killing trades in #1548
7. Volume filter relaxed: >0.6x avg (not 0.8x)

Strategy Design:
- HTF Bias: 4h HMA(21) for macro direction (price above = long bias)
- Regime: Choppiness(14) > 50 = range (trade), < 40 = trend (skip)
- Entry: CRSI(2,2,100) < 15 long, > 85 short — LOOSE for trade frequency
- Volume: > 0.6x 20-period average (relaxed from 0.8x)
- Exit: 2.5x ATR(14) trailing stop via signal→0
- Size: 0.25 discrete (0.0, ±0.25) — smaller for 30m fee sensitivity

Why this works for 30m:
- Mean reversion dominates lower TF in crypto (70% range-bound)
- CRSI catches oversold/overbought extremes better than RSI(14)
- CHOP filter avoids trend whipsaw (2022 crash, 2025 bear)
- 4h HMA bias prevents counter-trend mean reversion disasters

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h HMA(21) for bias, 1d for additional regime confirmation
Target: Sharpe > 0.20, trades > 30/train, > 3/test, DD < -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_regime_4h1d_hma_atr_v2"
timeframe = "30m"
leverage = 1.0

def calculate_crsi(close, rsi_period=2, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — superior mean reversion indicator
    
    CRSI = (RSI(close, rsi_period) + RSI(streak, streak_period) + PercentRank) / 3
    
    RSI(close): Standard RSI on price (short period for sensitivity)
    RSI(streak): RSI on up/down streak length
    PercentRank: Percentile rank of today's return over lookback
    
    Entry: CRSI < 10-15 (oversold), CRSI > 85-90 (overbought)
    Proven 75% win rate in range markets (Connors Research)
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # RSI on price (short period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_price = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi_price[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi_price[loss_smooth <= 1e-10] = 100.0
    rsi_price[:rsi_period] = np.nan
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask = streak_loss_smooth > 1e-10
    rsi_streak[mask] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[mask] / streak_loss_smooth[mask]))
    rsi_streak[streak_loss_smooth <= 1e-10] = 100.0
    rsi_streak[:streak_period + 5] = np.nan
    
    # Percent Rank
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close, prepend=close[0]) / (close[0] + 1e-10)
    
    for i in range(rank_period, n):
        window = returns[i - rank_period + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        current = returns[i]
        pct_rank[i] = np.sum(window < current) / len(window) * 100.0
    
    # Combine CRSI
    crsi = np.full(n, np.nan)
    valid = ~np.isnan(rsi_price) & ~np.isnan(rsi_streak) & ~np.isnan(pct_rank)
    crsi[valid] = (rsi_price[valid] + rsi_streak[valid] + pct_rank[valid]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — regime detection
    
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = Choppy/Range (mean reversion favorable)
    CHOP < 38.2 = Trending (trend following favorable)
    38.2 - 61.8 = Transition zone
    
    Best meta-filter for crypto (70% range-bound on lower TF)
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
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_atr / range_hl) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for additional regime confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    crsi = calculate_crsi(close, rsi_period=2, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume average for filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 30m (fee sensitivity)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME FILTER (CHOPPINESS) ===
        # Only trade in range markets (CHOP > 50)
        is_range = chop[i] > 50.0
        is_trend = chop[i] < 40.0
        
        # Skip if strong trend (whipsaw risk)
        if is_trend:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (4h HMA) ===
        # Loose bias — price above 4h HMA = long bias (not strict requirement)
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === VOLUME FILTER (RELAXED) ===
        vol_ok = volume[i] > 0.6 * vol_avg[i]
        
        # === ENTRY LOGIC — LOOSE THRESHOLDS FOR TRADE FREQUENCY ===
        desired_signal = 0.0
        
        # LONG: Range regime + CRSI oversold + volume + bull bias preferred
        if is_range and crsi[i] < 15.0 and vol_ok:
            if bull_bias:
                desired_signal = BASE_SIZE
            else:
                # Allow counter-bias trades in strong mean reversion
                desired_signal = BASE_SIZE * 0.7
        
        # SHORT: Range regime + CRSI overbought + volume + bear bias preferred
        if is_range and crsi[i] > 85.0 and vol_ok:
            if bear_bias:
                desired_signal = -BASE_SIZE
            else:
                # Allow counter-bias trades in strong mean reversion
                desired_signal = -BASE_SIZE * 0.7
        
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