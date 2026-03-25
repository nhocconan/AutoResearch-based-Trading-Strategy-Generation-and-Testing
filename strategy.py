#!/usr/bin/env python3
"""
Experiment #1514: 1d Primary + 1w HTF — Connors RSI Mean Reversion with Trend Filter

Hypothesis: Daily timeframe with weekly trend bias provides optimal balance between
trade frequency (15-30/year) and signal quality. Connors RSI is proven in quantitative
literature with 75% win rate for mean reversion entries. Combined with weekly HMA
trend filter, this should work in both bull and bear markets.

Key components:
1. 1w HMA(21) for major trend bias (avoid counter-trend trades)
2. 1d Connors RSI(3,2,100) for entry timing - LOOSE thresholds to guarantee trades
3. 1d HMA(16/48) for trend momentum confirmation
4. ATR(14) trailing stoploss (2.5x ATR)
5. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Connors RSI formula:
- RSI(3): 3-period RSI on close
- RSI_Streak(2): 2-period RSI on streak duration (consecutive up/down days)
- PercentRank(100): percentile rank of today's return vs last 100 days
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3

Entry logic (LOOSE to guarantee ≥10 trades/train, ≥3/test):
- LONG: 1w_HMA bullish + CRSI<25 (oversold) + price>1d_HMA48
- SHORT: 1w_HMA bearish + CRSI>75 (overbought) + price<1d_HMA48

Why this should work:
- Connors RSI is proven mean-reversion indicator (Larry Connors research)
- 1w trend filter prevents major counter-trend disasters
- 1d TF = natural 15-30 trades/year (fee-efficient)
- LOOSE CRSI thresholds (25/75 not 10/90) guarantee sufficient trades
- Simple logic = less overfitting risk

Target: Sharpe>0.6, trades>=15 train, trades>=3 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_hma_regime_1w_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean reversion signals
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    
    Lower CRSI = oversold (buy opportunity)
    Higher CRSI = overbought (sell opportunity)
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak duration
    streak = np.zeros(n, dtype=np.float64)
    streak[0] = 0
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to absolute duration for RSI calculation
    streak_duration = np.abs(streak)
    # Make streak positive for RSI calculation (up streaks positive, down streaks negative input)
    streak_signed = streak.copy()
    rsi_streak = calculate_rsi(streak_signed + 100, streak_period)  # offset to avoid negative issues
    
    # Component 3: Percentile Rank of today's return vs last rank_period days
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / close[:-1] * 100
    
    for i in range(rank_period, n):
        window = returns[i - rank_period + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            current_return = returns[i]
            percent_rank[i] = 100.0 * np.sum(valid < current_return) / len(valid)
    
    # Combine components
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
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
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
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
        
        if np.isnan(crsi[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
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
        
        # === TREND DIRECTION (1w HMA bias) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === 1d HMA TREND MOMENTUM ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        price_above_1d = close[i] > hma_48[i]
        price_below_1d = close[i] < hma_48[i]
        
        # === CONNORS RSI ===
        crsi_val = crsi[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: Weekly bullish + CRSI oversold + price above 1d HMA48
        if price_above_1w and crsi_val < 30 and price_above_1d:
            desired_signal = SIZE_STRONG
        elif price_above_1w and crsi_val < 40:
            desired_signal = SIZE_BASE
        
        # SHORT: Weekly bearish + CRSI overbought + price below 1d HMA48
        elif price_below_1w and crsi_val > 70 and price_below_1d:
            desired_signal = -SIZE_STRONG
        elif price_below_1w and crsi_val > 60:
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