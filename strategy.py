#!/usr/bin/env python3
"""
Experiment #1481: 15m Primary + 1h/4h/1d HTF — Connors RSI Mean Reversion with HTF Trend Filter

Hypothesis: 15m timeframe is underexplored (0 experiments). Connors RSI (CRSI) has proven
75% win rate for mean reversion in academic literature. Combined with HTF trend filter
(1d/4h HMA) for direction bias, this should capture intraday pullbacks in trend direction.

Key components:
1. 1d HMA(21) for major trend bias (only trade long if price > 1d HMA, short if <)
2. 4h HMA(21) for intermediate trend confirmation
3. 15m Connors RSI(3,2,100) for entry timing (CRSI < 15 long, > 85 short)
4. Session filter: 00-12 UTC preferred (London/NY overlap)
5. ATR(14) trailing stoploss (2.0x ATR)
6. Discrete sizing: 0.15-0.20 (smaller for 15m frequency)

Why 15m might work:
- Captures intraday mean reversion within HTF trend
- CRSI extremes happen frequently enough (40-80 trades/year target)
- HTF filter prevents counter-trend disasters
- Session filter reduces noise during low-volume periods

Entry logic (LOOSE to guarantee trades):
- LONG: 1d_HMA bullish + 4h_HMA bullish + CRSI < 20
- SHORT: 1d_HMA bearish + 4h_HMA bearish + CRSI > 80
- Exit: CRSI crosses back through 50 OR stoploss hit

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (lower than higher TFs due to frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_hma_1h4h1d_session_v1"
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

def calculate_percentile_rank(close, period=100):
    """Percentile Rank - percentage of values in lookback period below current value"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        if not np.isnan(close[i]):
            window = close[i - period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= period:
                count_below = np.sum(valid[:-1] < valid[-1])
                pr[i] = 100.0 * count_below / (len(valid) - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: Calculate consecutive up/down days, then RSI of that series
    - Up streak: count consecutive closes > previous close
    - Down streak: count consecutive closes < previous close (negative)
    """
    n = len(close)
    if n < max(rsi_period, streak_period, pr_period) + 5:
        return np.full(n, np.nan)
    
    # RSI(close, 3)
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if not np.isnan(close[i]) and not np.isnan(close[i-1]):
            if close[i] > close[i-1]:
                streak[i] = max(1, streak[i-1] + 1) if streak[i-1] >= 0 else 1
            elif close[i] < close[i-1]:
                streak[i] = min(-1, streak[i-1] - 1) if streak[i-1] <= 0 else -1
            else:
                streak[i] = streak[i-1]
    
    # RSI of streak (use absolute values for RSI calculation)
    streak_abs = np.abs(streak)
    rsi_streak = calculate_rsi(streak_abs, streak_period)
    
    # Percentile Rank
    pr = calculate_percentile_rank(close, pr_period)
    
    # Combine
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + pr[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 15m indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
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
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        is_preferred_session = 0 <= hour_utc <= 12
        
        # === TREND DIRECTION (HTF HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === CONNORS RSI ===
        crsi_val = crsi[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 4h bullish + CRSI oversold
        if price_above_1d and price_above_4h and crsi_val < 20:
            if is_preferred_session:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 1d bearish + 4h bearish + CRSI overbought
        elif price_below_1d and price_below_4h and crsi_val > 80:
            if is_preferred_session:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === EXIT SIGNAL (CRSI mean reversion) ===
        if in_position and position_side > 0 and crsi_val > 55:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_val < 45:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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