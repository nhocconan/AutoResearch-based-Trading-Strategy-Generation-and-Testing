#!/usr/bin/env python3
"""
Experiment #515: 6h Primary + 1d/1w HTF — Fisher Transform + Ehlers Regime

Hypothesis: 6h timeframe with dual HTF (1d + 1w) provides unique edge not yet explored.
Fisher Transform excels at catching reversals in bear/range markets (2022 crash, 2025 bear).
Ehlers Decision Point detects trend vs range more responsively than Choppiness Index.
Dual HTF alignment (1d primary, 1w confirmation) filters false signals without over-filtering.

Strategy logic:
1. 1w HMA(21) = weekly trend bias (strongest HTF filter, rarely used on 6h)
2. 1d HMA(21) = daily trend confirmation (primary HTF)
3. 6h Fisher Transform(9) = entry timing (crosses -1.5 long, +1.5 short)
4. 6h Ehlers Decision Point = regime filter (trend vs range detection)
5. 6h ATR(14)*2.5 stoploss on all positions
6. Relaxed HTF: 1d must agree, 1w adds conviction (not hard requirement)

Key differentiators from failed 6h strategies:
- Fisher Transform instead of RSI/CRSI (proven reversal detection in literature)
- Ehlers Decision Point instead of Choppiness (adaptive, less lag)
- Dual HTF with relaxed alignment (1d required, 1w optional boost)
- Conservative sizing (0.25-0.30) with discrete levels to minimize fee churn

Target: Sharpe>0.40 (beat current 6h best=0.399), trades>=60 train, trades>=8 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_ehlers_dual_htf_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Fisher Transform - normalizes price to Gaussian distribution
    Highlights turning points better than RSI (Ehlers, 2002)
    
    Formula:
    1. Calculate typical price: (H + L) / 2
    2. Normalize: (price - lowest) / (highest - lowest) * 1.998 + 0.001
    3. Fisher: 0.5 * ln((1 + normalized) / (1 - normalized))
    4. Signal line: 1-period lag of Fisher
    
    Entry: Fisher crosses above -1.5 (long) or below +1.5 (short)
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    signal = np.zeros(n)
    signal[:] = np.nan
    
    for i in range(period, n):
        highest = np.nanmax(typical[i-period+1:i+1])
        lowest = np.nanmin(typical[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            normalized = (typical[i] - lowest) / price_range * 1.998 + 0.001
            normalized = np.clip(normalized, -0.999, 0.999)  # Prevent log errors
            
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
            
            if i > period:
                signal[i] = fisher[i-1]
            else:
                signal[i] = fisher[i]
        else:
            fisher[i] = 0.0
            signal[i] = 0.0
    
    return fisher, signal

def calculate_ehlers_decision_point(close, period=14):
    """
    Ehlers Decision Point - detects trend vs range market
    Based on Ehlers' adaptive indicators research
    
    Formula:
    1. Calculate smoothed price (Ehlers super smoother)
    2. Calculate trend component (change over period)
    3. Calculate cycle component (sum of absolute changes)
    4. Decision Point = |trend| / (|trend| + cycle)
    
    DP > 0.55 = trending market
    DP < 0.45 = range-bound market
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    # Ehlers Super Smoother (2-pole lowpass filter)
    a1 = np.exp(-np.sqrt(2) * np.pi / period)
    b1 = 2.0 * a1 * np.cos(np.sqrt(2) * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1.0 - c2 - c3
    
    smooth = np.zeros(n)
    smooth[:] = np.nan
    
    for i in range(2, n):
        if i == 2:
            smooth[i] = (close[i] + close[i-1] + close[i-2]) / 3.0
        else:
            smooth[i] = c1 * (close[i] + close[i-1]) / 2.0 + c2 * smooth[i-1] + c3 * smooth[i-2]
    
    # Calculate trend component (difference from smoothed)
    trend = np.zeros(n)
    trend[:] = np.nan
    
    for i in range(period, n):
        if not np.isnan(smooth[i]) and not np.isnan(smooth[i-period]):
            trend[i] = smooth[i] - smooth[i-period]
    
    # Calculate cycle component (absolute changes)
    cycle = np.zeros(n)
    cycle[:] = np.nan
    
    for i in range(1, n):
        if not np.isnan(smooth[i]) and not np.isnan(smooth[i-1]):
            cycle[i] = abs(smooth[i] - smooth[i-1])
    
    # Sum cycle over period
    cycle_sum = pd.Series(cycle).rolling(window=period, min_periods=period).sum().values
    
    # Decision Point
    dp = np.zeros(n)
    dp[:] = np.nan
    
    for i in range(period, n):
        if not np.isnan(trend[i]) and not np.isnan(cycle_sum[i]):
            denominator = abs(trend[i]) + cycle_sum[i]
            if denominator > 1e-10:
                dp[i] = abs(trend[i]) / denominator
            else:
                dp[i] = 0.5
        else:
            dp[i] = 0.5
    
    return dp

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    dp = calculate_ehlers_decision_point(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    hma_6h = calculate_hma(close, period=21)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(dp[i]) or np.isnan(hma_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d primary, 1w confirmation) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong bias: both 1d and 1w agree
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        # Moderate bias: only 1d agrees
        htf_mod_bull = htf_1d_bull and not htf_1w_bear
        htf_mod_bear = htf_1d_bear and not htf_1w_bull
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === EHRLERS DECISION POINT (regime) ===
        trending = dp[i] > 0.55  # Trending market
        ranging = dp[i] < 0.45  # Range-bound market
        neutral = not trending and not ranging
        
        # === FISHER TRANSFORM SIGNALS ===
        # Crossover signals (more reliable than absolute levels)
        fisher_long_cross = False
        fisher_short_cross = False
        
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_signal[i]):
            # Fisher crosses above -1.5 from below
            fisher_long_cross = fisher[i] > -1.5 and fisher[i-1] <= -1.5
            # Fisher crosses below +1.5 from above
            fisher_short_cross = fisher[i] < 1.5 and fisher[i-1] >= 1.5
        
        # Fisher extreme levels (mean reversion entries)
        fisher_extreme_long = fisher[i] < -1.8
        fisher_extreme_short = fisher[i] > 1.8
        
        # Fisher recovery (turning from extreme)
        fisher_rising = fisher[i] > fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else False
        fisher_falling = fisher[i] < fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else False
        
        # === VOLATILITY FILTER ===
        atr_ratio = atr[i] / np.nanmean(atr[max(0,i-100):i]) if i >= 100 else 1.0
        vol_normal = atr_ratio < 3.0  # Avoid extreme vol spikes
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TRENDING REGIME: Follow HTF with Fisher crossover confirmation
        if trending and vol_normal:
            # Strong HTF + Fisher crossover = strong position
            if htf_strong_bull and hma_bull and fisher_long_cross:
                desired_signal = SIZE_STRONG
            elif htf_strong_bear and hma_bear and fisher_short_cross:
                desired_signal = -SIZE_STRONG
            # Moderate HTF + Fisher crossover = base position
            elif htf_mod_bull and hma_bull and fisher_long_cross:
                desired_signal = SIZE_BASE
            elif htf_mod_bear and hma_bear and fisher_short_cross:
                desired_signal = -SIZE_BASE
            # Pullback entries in trend (Fisher extreme + recovery)
            elif htf_strong_bull and hma_bear and fisher_extreme_long and fisher_rising:
                desired_signal = SIZE_BASE * 0.8
            elif htf_strong_bear and hma_bull and fisher_extreme_short and fisher_falling:
                desired_signal = -SIZE_BASE * 0.8
        
        # RANGING REGIME: Mean reversion with Fisher extremes
        if ranging and vol_normal:
            # Fisher extreme + HTF agreement = base position
            if fisher_extreme_long and htf_mod_bull:
                desired_signal = SIZE_BASE
            elif fisher_extreme_short and htf_mod_bear:
                desired_signal = -SIZE_BASE
            # Fisher recovery from extreme (no HTF required in range)
            elif fisher_extreme_long and fisher_rising:
                desired_signal = SIZE_BASE * 0.8
            elif fisher_extreme_short and fisher_falling:
                desired_signal = -SIZE_BASE * 0.8
        
        # NEUTRAL REGIME: Conservative entries only with strong HTF
        if neutral and vol_normal:
            if htf_strong_bull and fisher_extreme_long and fisher_rising:
                desired_signal = SIZE_BASE * 0.7
            elif htf_strong_bear and fisher_extreme_short and fisher_falling:
                desired_signal = -SIZE_BASE * 0.7
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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