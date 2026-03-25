#!/usr/bin/env python3
"""
Experiment #1639: 1h Primary + 4h/12h HTF — CRSI Mean Reversion with Trend Filter

Hypothesis: 1h timeframe with 4h trend bias + Connors RSI (CRSI) for entries captures
optimal mean-reversion opportunities while respecting higher timeframe direction.
CRSI has proven 75% win rate in research, especially effective in bear/range markets (2022-2024).

Key design choices based on failure analysis:
1. CRSI thresholds LOOSE: <25/>75 (not <10/>90) to GUARANTEE trades
2. 4h HMA(21) for trend bias - only filters direction, doesn't block entries
3. Choppiness Index for regime-aware sizing (not entry blocking)
4. Session filter REMOVED (was causing 0 trades in #1630, #1637)
5. Discrete signal sizes: 0.20 base, 0.30 strong conviction
6. 2.5x ATR trailing stoploss via signal→0
7. NO volume filter (too restrictive based on #1607, #1612 failures)

CRSI Formula (Connors RSI):
- RSI(3) of close
- RSI(2) of up/down streak length
- PercentRank(100) of today's return vs last 100 days
- CRSI = (RSI3 + RSI_Streak + PercentRank) / 3

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 4h HMA bullish + CRSI<25 + price>SMA200 (or just CRSI<20 if no SMA filter)
- SHORT: 4h HMA bearish + CRSI>75 + price<SMA200 (or just CRSI>80 if no SMA filter)
- Range regime (CHOP>61): Can enter against 4h trend with tighter CRSI

Why this beats recent failures:
- 1h TF = more responsive than 4h/6h, catches reversals faster
- CRSI proven superior to simple RSI for mean reversion
- Looser thresholds = more trades = better statistical edge
- 4h trend filter prevents counter-trend disasters but doesn't block all entries

Target: Sharpe>0.6, trades≥30 train, trades≥3 test, DD>-35%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h12h_loose_v1"
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
    Connors RSI (CRSI) - Combines 3 components for mean reversion signal
    1. RSI(3) of close price
    2. RSI(2) of up/down streak length
    3. PercentRank(100) of today's return vs last 100 periods
    
    CRSI < 10 = extremely oversold (long)
    CRSI > 90 = extremely overbought (short)
    We use looser thresholds (<25/>75) to guarantee trades
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan, dtype=np.float64)
    
    # Component 1: RSI(3) of close
    rsi3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak length
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 else -1
        else:
            streak[i] = streak[i-1] if i > 0 else 0
    
    # Convert streak to absolute value for RSI calculation
    streak_abs = np.abs(streak)
    # Create up/down for streak RSI
    streak_up = np.where(streak > 0, streak_abs, 0)
    streak_down = np.where(streak < 0, streak_abs, 0)
    
    avg_streak_up = pd.Series(streak_up).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_down = pd.Series(streak_down).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan, dtype=np.float64)
    mask = avg_streak_down != 0
    rs_streak = np.zeros(n)
    rs_streak[mask] = avg_streak_up[mask] / avg_streak_down[mask]
    rsi_streak[mask] = 100 - (100 / (1 + rs_streak[mask]))
    rsi_streak[avg_streak_down == 0] = 100  # All ups
    
    # Component 3: PercentRank of returns
    returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i-1] != 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100
    
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = returns[i - rank_period:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            count_below = np.sum(valid < returns[i])
            percent_rank[i] = count_below / len(valid) * 100
    
    # Combine all three components
    for i in range(rank_period, n):
        if not np.isnan(rsi3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
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
    min_bars = 250  # Need enough for CRSI rank_period + SMA200
    
    for i in range(min_bars, n):
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
        
        if np.isnan(sma_200[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === 12h HMA for stronger trend confirmation ===
        price_above_12h = close[i] > hma_12h_aligned[i] if not np.isnan(hma_12h_aligned[i]) else price_above_4h
        price_below_12h = close[i] < hma_12h_aligned[i] if not np.isnan(hma_12h_aligned[i]) else price_below_4h
        
        # === SMA200 for long-term bias ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CRSI SIGNALS (LOOSE thresholds for trades) ===
        crsi_val = crsi[i]
        crsi_oversold = crsi_val < 25  # LOOSE (was <10)
        crsi_overbought = crsi_val > 75  # LOOSE (was >90)
        crsi_extreme_oversold = crsi_val < 15
        crsi_extreme_overbought = crsi_val > 85
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow 4h trend with CRSI pullback entries
        if is_trend_regime:
            # LONG: 4h bullish + CRSI oversold pullback
            if price_above_4h and crsi_oversold:
                # Strong signal if 12h also bullish
                if price_above_12h:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: 4h bearish + CRSI overbought pullback
            elif price_below_4h and crsi_overbought:
                # Strong signal if 12h also bearish
                if price_below_12h:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion at extremes (can go against 4h trend)
        elif is_range_regime:
            # LONG: CRSI extreme oversold (stronger signal needed in range)
            if crsi_extreme_oversold:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI extreme overbought
            elif crsi_extreme_overbought:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Use 4h trend + SMA200 filter + moderate CRSI
        else:
            # LONG: 4h bullish + SMA200 support + CRSI not overbought
            if price_above_4h and price_above_sma200 and crsi_val < 60:
                if crsi_oversold:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: 4h bearish + SMA200 resistance + CRSI not oversold
            elif price_below_4h and price_below_sma200 and crsi_val > 40:
                if crsi_overbought:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === FALLBACK: Pure CRSI extremes (guarantees trades) ===
        # If no signal yet but CRSI at extreme, take the trade
        if desired_signal == 0.0:
            if crsi_extreme_oversold:
                desired_signal = SIZE_BASE
            elif crsi_extreme_overbought:
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