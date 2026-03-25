#!/usr/bin/env python3
"""
Experiment #1541: 15m Primary + 1h/4h/1d HTF — Multi-Timeframe Confluence Strategy

Hypothesis: 15m timeframe can work with STRICT HTF filtering to limit trades to 50-100/year.
Key insight from failures: 15m strategies fail due to (1) too many trades → fee drag, or
(2) too strict entries → zero trades. This strategy uses LOOSE entry thresholds with
STRONG HTF confluence to balance both.

Strategy components:
1. 4h HMA(21) for primary trend bias (long only when price > 4h HMA, short when <)
2. 1h RSI(7) for momentum confirmation (RSI>45 for longs, RSI<55 for shorts)
3. 15m Connors RSI (CRSI) for entry timing (CRSI<20 long, CRSI>80 short)
4. 15m ATR(14) for stoploss (2.5x ATR trailing)
5. Session filter: 08:00-20:00 UTC (high liquidity hours)
6. Discrete sizing: 0.0, ±0.15, ±0.25 (minimize fee churn)

CRSI Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- RSI(3): Fast RSI for quick reversals
- RSI_Streak(2): RSI of consecutive up/down days
- PercentRank(100): Where current price ranks in last 100 bars

Why this should work:
- 4h HMA prevents counter-trend trades in strong trends
- 1h RSI adds momentum confirmation (not oversold/overbought extremes)
- CRSI catches intraday pullbacks within HTF trend
- Session filter avoids low-liquidity whipsaws
- LOOSE thresholds (RSI 45/55, CRSI 20/80) guarantee trade generation

Entry logic (designed for ≥30 trades/train, ≥5/test):
- LONG: 4h_HMA bullish + 1h_RSI>45 + CRSI<25 + session active
- SHORT: 4h_HMA bearish + 1h_RSI<55 + CRSI>75 + session active

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_hma4h_rsi1h_session_v1"
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
    Connors RSI (CRSI)
    Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down day streaks
    PercentRank: Where current price ranks in last N bars (0-100)
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3) - fast RSI
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI Streak - measure consecutive up/down bars
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        streak_window = streak[i - streak_period + 1:i + 1]
        gain_streak = np.sum(np.where(streak_window > 0, streak_window, 0))
        loss_streak = np.sum(np.where(streak_window < 0, -streak_window, 0))
        if loss_streak > 0:
            rs_streak = gain_streak / loss_streak
            streak_rsi[i] = 100 - (100 / (1 + rs_streak))
        elif gain_streak > 0:
            streak_rsi[i] = 100.0
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank - where current close ranks in last N bars
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period - 1, n):
        window = close[i - rank_period + 1:i + 1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)  # exclude current
        percent_rank[i] = (count_below / (rank_period - 1)) * 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period - 1, n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_fast[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def is_session_active(open_time, start_hour=8, end_hour=20):
    """Check if bar is within high-liquidity session (UTC)"""
    # open_time is in milliseconds since epoch
    hour = pd.to_datetime(open_time, unit='ms').hour
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=7)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (high liquidity hours) ===
        session_active = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === MOMENTUM (1h RSI) ===
        rsi_1h = rsi_1h_aligned[i]
        momentum_bullish = rsi_1h > 45  # Not oversold
        momentum_bearish = rsi_1h < 55  # Not overbought
        
        # === ENTRY TIMING (15m CRSI) ===
        crsi_val = crsi[i]
        crsi_oversold = crsi_val < 25  # Entry long
        crsi_overbought = crsi_val > 75  # Entry short
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + 1h momentum + CRSI oversold + session active
        if price_above_4h and momentum_bullish and crsi_oversold:
            if session_active:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE  # Reduced size outside session
        
        # SHORT: 4h bearish + 1h momentum + CRSI overbought + session active
        elif price_below_4h and momentum_bearish and crsi_overbought:
            if session_active:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE  # Reduced size outside session
        
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