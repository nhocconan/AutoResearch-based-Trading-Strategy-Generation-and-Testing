#!/usr/bin/env python3
"""
Experiment #1450: 1h Primary + 4h/1d HTF — Choppiness Regime + cRSI Pullback

Hypothesis: 1h timeframe with HTF filters generates 40-80 trades/year (fee-efficient).
This strategy combines:
1. 1d HMA(21) for major trend bias (avoid counter-trend trades)
2. 4h Choppiness Index(14) for regime detection (trend vs range)
3. 1h Connors RSI (cRSI) for pullback entries (more responsive than standard RSI)
4. Session filter (08-20 UTC) to avoid low-liquidity hours
5. ATR(14) trailing stoploss (signal→0 when stopped)
6. Discrete sizing: 0.0, ±0.20, ±0.30 (minimize fee churn)

Why this should work:
- Choppiness Index filters out whipsaw range markets (CHOP > 61.8 = avoid trend trades)
- cRSI is more responsive than RSI(14) for pullback detection
- 1h TF = natural 40-80 trades/year (not overtraded like 15m/30m)
- Session filter reduces noise during Asian low-volume hours
- LOOSE entry thresholds guarantee trades (cRSI < 50 for long, > 50 for short)

Entry logic (LOOSE to guarantee trades):
- LONG: 1d_HMA bullish + 4h_CHOP < 55 + cRSI < 50 + session 08-20 UTC
- SHORT: 1d_HMA bearish + 4h_CHOP < 55 + cRSI > 50 + session 08-20 UTC

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_hma_session_4h1d_v1"
timeframe = "1h"
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
    PercentRank: percentage of recent returns lower than current return
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI of streak length
    streak = np.zeros(n, dtype=np.float64)
    streak_direction = np.zeros(n, dtype=np.float64)  # +1 for up, -1 for down
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streak_direction[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
            streak_direction[i] = 1
        elif close[i] < close[i-1]:
            if streak_direction[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
            streak_direction[i] = -1
        else:
            streak[i] = 0
            streak_direction[i] = 0
    
    # Convert streak to RSI-like value (map to 0-100)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period+1:i+1]
        if len(streak_window) == streak_period:
            # Simple mapping: positive streak = bullish, negative = bearish
            avg_streak = np.mean(streak_window)
            streak_rsi[i] = 50 + avg_streak * 10  # Scale to reasonable range
            streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    # Percent Rank of recent returns
    returns = np.diff(close) / close[:-1]
    returns = np.insert(returns, 0, 0)
    
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if len(window) == rank_period and not np.any(np.isnan(window)):
            count_lower = np.sum(window[:-1] < window[-1])
            percent_rank[i] = count_lower / (rank_period - 1) * 100
    
    # Combine into cRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(atr[i]):
            atr_sum = np.nansum(atr[i-period+1:i+1])
            highest_high = np.nanmax(high[i-period+1:i+1])
            lowest_low = np.nanmin(low[i-period+1:i+1])
            price_range = highest_high - lowest_low
            
            if price_range > 1e-10 and atr_sum > 0:
                chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
                chop[i] = np.clip(chop[i], 0, 100)
    
    return chop

def is_session_active(open_time_unix, start_hour=8, end_hour=20):
    """Check if timestamp is within active trading session (UTC)"""
    # Convert unix ms to hour
    hour = (open_time_unix // (1000 * 60 * 60)) % 24
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    chop_4h_raw = calculate_choppiness(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate 1h indicators
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        
        if np.isnan(crsi_1h[i]):
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
        
        if np.isnan(chop_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME FILTER (4h Choppiness) ===
        chop = chop_4h_aligned[i]
        is_trending = chop < 55  # Below 55 = trending (loose threshold)
        is_choppy = chop > 61.8  # Above 61.8 = range (avoid trend trades)
        
        # === ENTRY SIGNAL (cRSI pullback - LOOSE thresholds) ===
        crsi = crsi_1h[i]
        
        # === SESSION FILTER (08-20 UTC) ===
        session_active = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + trending regime + cRSI pullback + session active
        if price_above_1d and is_trending and crsi < 50 and session_active:
            # Strong if cRSI < 35 (deeper pullback)
            if crsi < 35:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 1d bearish + trending regime + cRSI pullback + session active
        elif price_below_1d and is_trending and crsi > 50 and session_active:
            # Strong if cRSI > 65 (deeper pullback)
            if crsi > 65:
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