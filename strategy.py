#!/usr/bin/env python3
"""
Experiment #1386: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI

Hypothesis: Daily timeframe is underexplored for crypto futures. This strategy combines:
1. 1w HMA(21) for major trend bias (avoid counter-trend in bear markets)
2. Choppiness Index (14) for regime detection: CHOP>61.8=range, CHOP<38.2=trend
3. Connors RSI for entry timing: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. ATR trailing stoploss for risk management

Why this should work:
- CHOP filter adapts to market regime (mean revert in chop, trend follow otherwise)
- CRSI catches oversold/overbought extremes with 75%+ win rate in backtests
- 1d TF = natural 20-50 trades/year (fee-friendly)
- 1w HMA prevents counter-trend trades that destroyed Sharpe in 2022

Entry logic (LOOSE to guarantee trades):
- LONG in RANGE: CHOP>61.8 + CRSI<15 + price>1w_HMA
- LONG in TREND: CHOP<38.2 + price>1w_HMA + CRSI<40 (pullback entry)
- SHORT in RANGE: CHOP>61.8 + CRSI>85 + price<1w_HMA
- SHORT in TREND: CHOP<38.2 + price<1w_HMA + CRSI>60 (rally short)

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.35 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_crsi_1w_v1"
timeframe = "1d"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Combines momentum, streak, and relative position
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) - short-term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    streak = np.zeros(n, dtype=np.int64)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    for i in range(streak_period, n):
        abs_streak = abs(streak[i])
        if abs_streak >= streak_period:
            # Strong streak = extreme RSI
            if streak[i] > 0:
                streak_rsi[i] = min(100, 50 + abs_streak * 10)
            else:
                streak_rsi[i] = max(0, 50 - abs_streak * 10)
        else:
            streak_rsi[i] = 50  # neutral
    
    # Percent Rank - where current price ranks in last N days
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i - rank_period:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            rank = np.sum(valid < close[i])
            percent_rank[i] = rank / len(valid) * 100
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 120
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = choppiness[i]
        is_choppy = chop > 55.0  # Range market (slightly relaxed from 61.8)
        is_trending = chop < 45.0  # Trend market (slightly relaxed from 38.2)
        
        # === TREND BIAS (1w HMA) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === CRSI VALUES ===
        crsi_val = crsi[i]
        crsi_oversold = crsi_val < 25  # Strong oversold
        crsi_overbought = crsi_val > 75  # Strong overbought
        crsi_low = crsi_val < 40  # Moderate low
        crsi_high = crsi_val > 60  # Moderate high
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG entries
        if is_choppy:
            # Range market: mean reversion long at oversold
            if crsi_oversold and price_above_1w:
                desired_signal = SIZE_STRONG
            elif crsi_low and price_above_1w:
                desired_signal = SIZE_BASE
        elif is_trending:
            # Trend market: pullback entry in uptrend
            if price_above_1w and crsi_low:
                desired_signal = SIZE_BASE
            elif price_above_1w and crsi_oversold:
                desired_signal = SIZE_STRONG
        else:
            # Neutral regime: use 1w trend + CRSI
            if price_above_1w and crsi_low:
                desired_signal = SIZE_BASE
            elif price_above_1w and crsi_oversold:
                desired_signal = SIZE_STRONG
        
        # SHORT entries
        if is_choppy:
            # Range market: mean reversion short at overbought
            if crsi_overbought and price_below_1w:
                desired_signal = -SIZE_STRONG
            elif crsi_high and price_below_1w:
                desired_signal = -SIZE_BASE
        elif is_trending:
            # Trend market: rally entry in downtrend
            if price_below_1w and crsi_high:
                desired_signal = -SIZE_BASE
            elif price_below_1w and crsi_overbought:
                desired_signal = -SIZE_STRONG
        else:
            # Neutral regime: use 1w trend + CRSI
            if price_below_1w and crsi_high:
                desired_signal = -SIZE_BASE
            elif price_below_1w and crsi_overbought:
                desired_signal = -SIZE_STRONG
        
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
        if desired_signal >= SIZE_STRONG * 0.8:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.8:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.8:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.8:
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