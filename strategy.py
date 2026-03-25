#!/usr/bin/env python3
"""
Experiment #1242: 4h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 1000+ failed experiments, the winning pattern is:
1. Higher timeframe (4h) for natural 20-50 trades/year (fee-efficient)
2. Daily HMA for clear trend direction (simple, proven)
3. Connors RSI for superior mean-reversion entry timing (75% win rate in literature)
4. Choppiness Index to avoid whipsaw in range markets (critical for 2022-2024)
5. LOOSE entry thresholds to guarantee trades (CRSI < 25 or > 75, not extreme 10/90)

Key insight: Most failures come from over-filtered entries (0 trades) or trading in chop.
This strategy uses Choppiness > 50 to confirm range (mean revert) vs < 50 for trend follow.
Connors RSI captures short-term oversold/overbought within the larger trend.

Entry logic:
- LONG: price > 1d_HMA AND CHOP > 50 (range) AND CRSI < 25 (oversold pullback)
- LONG: price > 1d_HMA AND CHOP < 50 (trend) AND CRSI < 35 (trend pullback)
- SHORT: price < 1d_HMA AND CHOP > 50 (range) AND CRSI > 75 (overbought pullback)
- SHORT: price < 1d_HMA AND CHOP < 50 (trend) AND CRSI > 65 (trend pullback)

Why this should beat current best (Sharpe=0.445):
- 4h TF = optimal trade frequency (less noise than 1h, more signals than 6h/12h)
- Choppiness filter = avoids 2022-style whipsaw destruction
- Connors RSI = better entry timing than standard RSI(14)
- Loose thresholds = guarantees 30+ trades on train, 5+ on test

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_hma_regime_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3) - fast RSI for short-term extremes
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI of Streak - measure consecutive up/down days
    streak = np.zeros(n, dtype=np.float64)
    streak[0] = 1
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rs_streak = np.divide(avg_streak_gain, avg_streak_loss, out=np.ones_like(avg_streak_gain), where=avg_streak_loss != 0)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak[:streak_period] = np.nan
    
    # Percent Rank - today's return vs last 100 days
    returns = np.diff(close, prepend=close[0]) / np.where(close != 0, close, 1)
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(rank_period, n):
        window = returns[i - rank_period:i]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window < returns[i])
            percent_rank[i] = (count_below / rank_period) * 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_fast[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return choppiness

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (Daily HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # Weekly HMA for additional confirmation
        hma_1w_valid = not np.isnan(hma_1w_aligned[i])
        price_above_1w = hma_1w_valid and close[i] > hma_1w_aligned[i]
        price_below_1w = hma_1w_valid and close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop = choppiness[i]
        is_choppy = chop > 50.0  # Range market - use mean reversion
        is_trending = chop <= 50.0  # Trending market - use trend pullback
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        crsi_val = crsi[i]
        
        # LONG entries
        if price_above_1d:
            if is_choppy:
                # Range market: mean reversion at oversold
                if crsi_val < 25.0:
                    if price_above_1w:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            else:
                # Trending market: pullback in uptrend
                if crsi_val < 35.0:
                    if price_above_1w:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
        
        # SHORT entries
        elif price_below_1d:
            if is_choppy:
                # Range market: mean reversion at overbought
                if crsi_val > 75.0:
                    if price_below_1w:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
            else:
                # Trending market: pullback in downtrend
                if crsi_val > 65.0:
                    if price_below_1w:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals