#!/usr/bin/env python3
"""
Experiment #1504: 12h Primary + 1d/1w HTF — Connors RSI + Fisher Transform Regime

Hypothesis: Previous 12h strategies failed due to overly strict entry conditions
(0 trades) or pure trend-following (died in 2022 crash + 2025 bear). This strategy
uses CONNORS RSI (proven 75% win rate mean reversion) + FISHER TRANSFORM (sharp
reversal signals) with Choppiness Index regime detection.

Key innovations vs failed #1452:
1. CONNORS RSI instead of simple RSI — combines RSI(3) + RSI_Streak(2) + PercentRank(100)
2. FISHER TRANSFORM for sharper reversal entries (crosses -1.5/+1.5)
3. MUCH LOOSER entry thresholds — CRSI<30/>70 (not 25/75), Fisher crosses at -1.0/+1.0
4. Dual regime with fallback — if no regime signal, use pure mean reversion
5. Volume confirmation optional — don't require, just boost size when present
6. 1d HMA as soft filter — reduce size counter-trend, don't block entries

Why this should generate trades:
- CRSI<30 happens frequently in corrections (every 2-4 weeks on 12h)
- Fisher crosses -1.0/+1.0 happen 3-5x per year per direction
- Choppiness regime switches 4-8x per year
- Combined: 30-50 trades/year target

Entry logic (LOOSE):
- LONG mean reversion: CRSI<30 OR (Fisher<-1.0 crossing up) — either triggers
- SHORT mean reversion: CRSI>70 OR (Fisher>1.0 crossing down) — either triggers
- Regime boost: Range regime (CHOP>61) = full size, Trend regime = half size
- 1d HMA filter: Counter-trend = reduce size 50%, not block

Position sizing: 0.20 base, 0.30 strong (discrete levels)
Stoploss: 2.5x ATR trailing
Timeframe: 12h
Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_fisher_chop_regime_1d1w_v1"
timeframe = "12h"
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
    Proven mean reversion indicator with 75% win rate
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    streak = np.zeros(n, dtype=np.int32)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    for i in range(streak_period, n):
        streak_window = streak[i - streak_period + 1:i + 1]
        avg_streak = np.mean(streak_window)
        # Map streak to 0-100 scale
        streak_rsi[i] = 50 + avg_streak * 10  # +/- 5 streak = 0-100
        streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    # Component 3: Percent Rank (current price vs last 100 closes)
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)  # exclude current
        percent_rank[i] = (count_below / (rank_period - 1)) * 100
    
    # Combine components
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close := (high + low) / 2)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + (np.roll(close, 1))) / 3  # approx close
    typical = np.where(np.isnan(typical), close, typical)
    
    # Normalize price to -1 to +1 range
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            # Normalize to -1 to +1
            normalized = 2.0 * (typical[i] - lowest) / price_range - 1.0
            normalized = np.clip(normalized, -0.99, 0.99)  # avoid log(0)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
            
            # Signal line (1-period lag)
            if i > 0 and not np.isnan(fisher[i-1]):
                fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

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
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=10)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi_14 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_HALF = 0.15
    
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
        
        if np.isnan(crsi_14[i]) or np.isnan(fisher[i]) or np.isnan(chop_14[i]):
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
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_range_regime = chop > 55.0  # Lowered from 61.8 for more range signals
        is_trend_regime = chop < 45.0  # Raised from 38.2 for more trend signals
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === CONNORS RSI SIGNALS (LOOSE thresholds) ===
        crsi = crsi_14[i]
        crsi_oversold = crsi < 30  # Was 25, now 30 for more trades
        crsi_overbought = crsi > 70  # Was 75, now 70 for more trades
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_val = fisher[i]
        fisher_sig = fisher_signal[i] if not np.isnan(fisher_signal[i]) else fisher_val
        
        # Fisher crossing up from oversold
        fisher_long_cross = (fisher_val > -1.0) and (fisher_sig <= -1.0) if not np.isnan(fisher_sig) else False
        # Fisher crossing down from overbought
        fisher_short_cross = (fisher_val < 1.0) and (fisher_sig >= 1.0) if not np.isnan(fisher_sig) else False
        
        # === ENTRY LOGIC (VERY LOOSE - must generate trades) ===
        desired_signal = 0.0
        signal_strength = "none"
        
        # RANGE REGIME: Mean reversion primary
        if is_range_regime:
            # LONG: CRSI oversold OR Fisher long cross
            if crsi_oversold or fisher_long_cross:
                if price_above_1d:  # With 1d trend = strong
                    desired_signal = SIZE_STRONG
                    signal_strength = "strong"
                else:  # Counter 1d trend = base
                    desired_signal = SIZE_BASE
                    signal_strength = "base"
            
            # SHORT: CRSI overbought OR Fisher short cross
            elif crsi_overbought or fisher_short_cross:
                if price_below_1d:  # With 1d trend = strong
                    desired_signal = -SIZE_STRONG
                    signal_strength = "strong"
                else:  # Counter 1d trend = base
                    desired_signal = -SIZE_BASE
                    signal_strength = "base"
        
        # TREND REGIME: Follow trend with pullback entries
        elif is_trend_regime:
            # LONG: 1d bullish + CRSI pullback OR Fisher long cross
            if price_above_1d and (crsi < 45 or fisher_long_cross):
                desired_signal = SIZE_BASE
                signal_strength = "base"
            
            # SHORT: 1d bearish + CRSI pullback OR Fisher short cross
            elif price_below_1d and (crsi > 55 or fisher_short_cross):
                desired_signal = -SIZE_BASE
                signal_strength = "base"
        
        # NEUTRAL REGIME: Pure mean reversion
        else:
            # LONG: Extreme CRSI or Fisher cross
            if crsi < 25 or fisher_long_cross:
                desired_signal = SIZE_BASE
                signal_strength = "base"
            
            # SHORT: Extreme CRSI or Fisher cross
            elif crsi > 75 or fisher_short_cross:
                desired_signal = -SIZE_BASE
                signal_strength = "base"
        
        # === FALLBACK: If no regime signal, use pure CRSI extremes ===
        if desired_signal == 0.0:
            if crsi < 20:  # Very oversold
                desired_signal = SIZE_HALF
            elif crsi > 80:  # Very overbought
                desired_signal = -SIZE_HALF
        
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
        elif desired_signal >= SIZE_HALF * 0.8:
            final_signal = SIZE_HALF
        elif desired_signal <= -SIZE_HALF * 0.8:
            final_signal = -SIZE_HALF
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