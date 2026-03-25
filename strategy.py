#!/usr/bin/env python3
"""
Experiment #1430: 1h Primary + 4h/1d HTF — Choppiness + cRSI + Session Filter

Hypothesis: 1h timeframe with strict session + regime filters will generate 40-80 trades/year.
Combining:
1. 1d HMA(21) for major trend bias (avoid counter-trend)
2. 4h HMA(16/48) crossover for momentum confirmation
3. 1h Choppiness Index(14) for regime detection (>55 = range, <45 = trend)
4. 1h cRSI(3,2,100) for precise pullback entries
5. Session filter: 08-20 UTC only (avoid low-volume Asian session whipsaws)
6. ATR(14) trailing stoploss at 2.5x

Why this should work:
- Session filter reduces false signals during low-volume periods
- Choppiness filter adapts to market regime (trend vs range)
- cRSI is more responsive than standard RSI for entry timing
- 1h TF with HTF confirmation = fewer trades than pure 1h, more than 4h
- Discrete sizing minimizes fee churn

Entry logic (balanced - not too strict, not too loose):
- LONG: 1d_HMA bullish + 4h_HMA16>48 + CHOP<55 + cRSI<25 + session 08-20 UTC
- SHORT: 1d_HMA bearish + 4h_HMA16<48 + CHOP<55 + cRSI>75 + session 08-20 UTC

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_session_4h1d_v1"
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
    """Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n, dtype=np.float64)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    current_streak = 0
    for i in range(1, n):
        if delta[i] > 0:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
        elif delta[i] < 0:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
        else:
            current_streak = 0
        streak[i] = current_streak
    
    # RSI of streak
    streak_rsi = calculate_rsi(np.abs(streak), streak_period)
    # Adjust sign based on streak direction
    streak_rsi = np.where(streak >= 0, streak_rsi, 100 - streak_rsi)
    
    # Percent Rank (100)
    pct_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(close[i]):
            window = close[i - rank_period:i]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                count_below = np.sum(valid < close[i])
                pct_rank[i] = 100.0 * count_below / len(valid)
    
    # Combine
    crsi = np.full(n, np.nan, dtype=np.float64)
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(pct_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + pct_rank[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures if market is trending or ranging"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        if highest == lowest or np.isnan(highest) or np.isnan(lowest):
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

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
    
    hma_4h_16_raw = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_48_raw = calculate_hma(df_4h['close'].values, period=48)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16_raw)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    
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
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA CROSSOVER (trend momentum) ===
        hma_bullish = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_bearish = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === CHOPPINESS REGIME (<55 = trending, >=55 = ranging) ===
        is_trending = chop[i] < 55.0
        
        # === cRSI ENTRY (extreme values for pullback) ===
        crsi_value = crsi[i]
        crsi_oversold = crsi_value < 25.0
        crsi_overbought = crsi_value > 75.0
        
        # === ENTRY LOGIC (must generate trades - not too strict) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 4h HMA bullish + trending + cRSI oversold + session
        if in_session and price_above_1d and hma_bullish and is_trending and crsi_oversold:
            desired_signal = SIZE_STRONG
        
        # SHORT: 1d bearish + 4h HMA bearish + trending + cRSI overbought + session
        elif in_session and price_below_1d and hma_bearish and is_trending and crsi_overbought:
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