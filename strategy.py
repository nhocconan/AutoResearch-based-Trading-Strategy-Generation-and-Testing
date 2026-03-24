#!/usr/bin/env python3
"""
Experiment #916: 30m Primary + 4h/1d HTF — HMA Trend + Connors RSI + Choppiness Regime

Hypothesis: 30m timeframe with 4h/1d HTF bias provides optimal balance between
trade frequency (40-80/year) and signal quality. Lower TF than 6h/12h strategies
but using HTF for direction prevents overtrading. Key insight from failed 1h/15m
strategies: entry conditions must be LOOSE to ensure trades happen.

Key innovations:
1. 1d HMA(21) for highest-level trend bias (directional filter only)
2. 4h HMA(16/48) for intermediate trend confirmation
3. 30m Connors RSI for entry timing (loose thresholds: <40/>60 not just <25/>75)
4. Choppiness Index(14) regime: >50 = range (mean revert), <50 = trend follow
5. Session filter: 08-20 UTC only (major trading hours, reduces noise)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.20, ±0.30

CRITICAL LESSON FROM FAILURES (#905, #909, #910, #911, #913 = 0 trades):
- Entry conditions MUST be loose enough to generate trades
- HTF trend = bias only, not hard requirement
- CRSI thresholds widened: <40 for long, >60 for short (not just extremes)
- Session filter is permissive (12 hours, not 4 hours)
- At least ONE of: HTF bull + CRSI low OR HMA crossover + CRSI low

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 30m
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_crsi_chop_regime_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    if sqrt_n < 1:
        sqrt_n = 1
    
    def wma(series, span):
        if span < 1:
            span = 1
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    rsi_close = calculate_rsi(close, rsi_period)
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period, n):
        if avg_streak_loss[i] > 1e-10:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_streak[i] = 100.0
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        changes = np.diff(close[i - rank_period:i + 1])
        current_change = changes[-1]
        count_below = np.sum(changes[:-1] < current_change)
        percent_rank[i] = count_below / (rank_period - 1) * 100.0
    
    crsi = np.full(n, np.nan)
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 50 as threshold for simplicity
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def get_hour_from_open_time(open_time_arr):
    """Extract hour from open_time (milliseconds timestamp)"""
    hours = np.zeros(len(open_time_arr), dtype=np.int32)
    for i in range(len(open_time_arr)):
        ts_ms = open_time_arr[i]
        ts_s = ts_ms / 1000.0
        hours[i] = int((ts_s % 86400) / 3600)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_4h_16_raw = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_48_raw = calculate_hma(df_4h['close'].values, period=48)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16_raw)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 30m indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Session hours (08-20 UTC)
    hours = get_hour_from_open_time(open_time)
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
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
        
        # === HTF BIAS (1d HMA) - DIRECTIONAL FILTER ONLY ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA TREND ===
        hma_4h_bull = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bear = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === 4h HMA CROSSOVER (for additional confirmation) ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_4h_16_aligned[i-1]) and not np.isnan(hma_4h_48_aligned[i-1]):
            hma_crossover_long = (hma_4h_16_aligned[i-1] <= hma_4h_48_aligned[i-1]) and (hma_4h_16_aligned[i] > hma_4h_48_aligned[i])
            hma_crossover_short = (hma_4h_16_aligned[i-1] >= hma_4h_48_aligned[i-1]) and (hma_4h_16_aligned[i] < hma_4h_48_aligned[i])
        
        # === CRSI CONDITIONS (LOOSE THRESHOLDS FOR TRADES) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_low = crsi[i] < 40.0  # Loose long entry
        crsi_high = crsi[i] > 60.0  # Loose short entry
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 50.0
        chop_ranging = chop_14[i] >= 50.0
        
        # === SESSION FILTER ===
        session_ok = in_session[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS TO ENSURE TRADES) ===
        desired_signal = 0.0
        
        # LONG entries - multiple paths to ensure trades happen
        long_score = 0
        
        # Path 1: HTF bull + 4h bull + CRSI low (trend follow)
        if htf_1d_bull and hma_4h_bull and crsi_low:
            long_score += 2
        
        # Path 2: HTF bull + CRSI oversold (mean reversion in uptrend)
        if htf_1d_bull and crsi_oversold:
            long_score += 2
        
        # Path 3: 4h crossover long + CRSI low
        if hma_crossover_long and crsi_low:
            long_score += 2
        
        # Path 4: Range regime + HTF bull + CRSI low
        if chop_ranging and htf_1d_bull and crsi_low:
            long_score += 1
        
        # Path 5: Trend regime + 4h bull + CRSI low
        if chop_trending and hma_4h_bull and crsi_low:
            long_score += 1
        
        # SHORT entries - multiple paths
        short_score = 0
        
        # Path 1: HTF bear + 4h bear + CRSI high (trend follow)
        if htf_1d_bear and hma_4h_bear and crsi_high:
            short_score += 2
        
        # Path 2: HTF bear + CRSI overbought (mean reversion in downtrend)
        if htf_1d_bear and crsi_overbought:
            short_score += 2
        
        # Path 3: 4h crossover short + CRSI high
        if hma_crossover_short and crsi_high:
            short_score += 2
        
        # Path 4: Range regime + HTF bear + CRSI high
        if chop_ranging and htf_1d_bear and crsi_high:
            short_score += 1
        
        # Path 5: Trend regime + 4h bear + CRSI high
        if chop_trending and hma_4h_bear and crsi_high:
            short_score += 1
        
        # Determine signal based on scores (need score >= 2 for entry)
        if long_score >= 2 and short_score < 2:
            if long_score >= 4:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        elif short_score >= 2 and long_score < 2:
            if short_score >= 4:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        elif long_score >= 2 and short_score >= 2:
            # Conflicting signals - stay flat or follow HTF
            if htf_1d_bull:
                desired_signal = SIZE_BASE if long_score >= short_score else 0.0
            elif htf_1d_bear:
                desired_signal = -SIZE_BASE if short_score >= long_score else 0.0
        
        # Apply session filter (reduce size outside session, don't block entirely)
        if not session_ok and desired_signal != 0.0:
            desired_signal = desired_signal * 0.5  # Half size outside session
        
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