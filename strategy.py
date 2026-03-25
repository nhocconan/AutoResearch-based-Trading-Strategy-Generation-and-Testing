#!/usr/bin/env python3
"""
Experiment #1549: 15m Primary + 1h/1d HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: 15m timeframe is underexplored (0 experiments). Using Connors RSI (CRSI)
for mean-reversion entries with HTF regime filter should work because:
1. CRSI has 75% win rate in quantitative literature for short-term reversals
2. 1d HMA prevents counter-trend disasters in strong trends
3. 1h Choppiness Index filters out choppy periods where mean-reversion fails
4. Session filter (00-12 UTC) captures London/NY overlap liquidity
5. 15m provides precise entry timing while HTF controls direction

Key components:
1. 1d HMA(21) for major trend bias (long-only above, short-only below)
2. 1h Choppiness Index(14) for regime: CHOP<50 = trend-follow, CHOP>50 = mean-revert
3. 15m Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. Session filter: 00-12 UTC preferred (higher volume, cleaner signals)
5. ATR(14) trailing stoploss (2.5x ATR)
6. Discrete sizing: 0.0, ±0.15, ±0.20 (15m needs smaller size due to frequency)

Why this should work on 15m:
- CRSI extremes (<10 long, >90 short) are rare = selective entries
- HTF filters prevent trading against major trend
- Session filter cuts noise from low-volume Asian session
- Smaller position size (0.15-0.20) accounts for higher trade frequency
- Stoploss via signal→0 when 2.5*ATR hit

Entry logic (LOOSE to guarantee ≥40 trades/train, ≥5/test):
- LONG: 1d_HMA bullish + (CHOP<50 OR CRSI<15) + session 00-12 UTC
- SHORT: 1d_HMA bearish + (CHOP<50 OR CRSI>85) + session 00-12 UTC

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_chop_session_1h1d_v1"
timeframe = "15m"
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
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak length
    streak = np.zeros(n, dtype=np.float64)
    streak[0] = 1
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (absolute streak length)
    streak_abs = np.abs(streak)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        streak_window = streak_abs[i - streak_period:i + 1]
        if len(streak_window) == streak_period and np.all(streak_window >= 0):
            # Simple normalization: map streak 0-10 to 0-100
            avg_streak = np.mean(streak_window)
            streak_rsi[i] = min(100, avg_streak * 10)
    
    # Component 3: Percentile Rank of returns
    returns = np.diff(close) / close[:-1]
    returns = np.insert(returns, 0, 0)
    
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = returns[i - rank_period:i]
        if len(window) == rank_period and not np.any(np.isnan(window)):
            count_below = np.sum(window < returns[i])
            percent_rank[i] = count_below / rank_period * 100
    
    # Combine components
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    chop_1h_raw = calculate_choppiness(
        df_1h['high'].values,
        df_1h['low'].values,
        df_1h['close'].values,
        period=14
    )
    chop_1h_aligned = align_htf_to_ltf(prices, df_1h, chop_1h_raw)
    
    # Calculate 15m indicators
    crsi_15m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
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
        
        if np.isnan(crsi_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        utc_hour = get_utc_hour(open_time[i])
        is_preferred_session = 0 <= utc_hour <= 12
        
        # === REGIME DETECTION (1h Choppiness) ===
        chop = chop_1h_aligned[i]
        is_trend_regime = chop < 50.0
        is_range_regime = chop > 50.0
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi = crsi_15m[i]
        crsi_oversold = crsi < 15  # Strong mean-reversion long signal
        crsi_overbought = crsi > 85  # Strong mean-reversion short signal
        crsi_moderate_oversold = crsi < 25
        crsi_moderate_overbought = crsi > 75
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG entries
        if price_above_1d:  # Only long when above 1d HMA
            if is_preferred_session:
                # Strong signal: CRSI extreme + any regime
                if crsi_oversold:
                    desired_signal = SIZE_STRONG
                # Moderate signal: CRSI moderate + trend regime
                elif crsi_moderate_oversold and is_trend_regime:
                    desired_signal = SIZE_BASE
                # Weak signal: CRSI moderate + range regime
                elif crsi_moderate_oversold and is_range_regime:
                    desired_signal = SIZE_BASE * 0.5
        
        # SHORT entries
        elif price_below_1d:  # Only short when below 1d HMA
            if is_preferred_session:
                # Strong signal: CRSI extreme + any regime
                if crsi_overbought:
                    desired_signal = -SIZE_STRONG
                # Moderate signal: CRSI moderate + trend regime
                elif crsi_moderate_overbought and is_trend_regime:
                    desired_signal = -SIZE_BASE
                # Weak signal: CRSI moderate + range regime
                elif crsi_moderate_overbought and is_range_regime:
                    desired_signal = -SIZE_BASE * 0.5
        
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
        elif desired_signal >= SIZE_BASE * 0.4:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_BASE * 0.4:
            final_signal = -SIZE_BASE * 0.5
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