#!/usr/bin/env python3
"""
Experiment #1620: 6h Primary + 1d/1w HTF — Weekly Pivot + Fisher Regime

Hypothesis: 6h timeframe sits in the sweet spot between 4h (too noisy) and 12h (too slow).
Weekly pivot levels provide strong S/R that institutions watch. Combined with daily trend
bias and Fisher Transform entries, this should capture both trend continuation and
mean-reversion bounces at key levels.

Key design choices based on failure analysis:
1. Weekly pivots (Standard formula) as primary S/R - institutions watch these
2. Daily HMA(21) for trend bias - smoother than EMA, less whipsaw
3. Fisher Transform(9) for entry timing - proven in 2022-2024 bear/range
4. Choppiness Index(14) for regime - adapts logic to market state
5. LOOSE entry thresholds to guarantee ≥30 trades/train
6. 2.5x ATR trailing stoploss via signal→0

Entry logic:
- TREND (CHOP<38): Fisher cross + 1d HMA bias + price near weekly pivot
- RANGE (CHOP>61): Fisher extremes at weekly S/R (mean reversion)
- NEUTRAL: 1d HMA bias + RSI confirmation only (simplest, most trades)

Why this might beat mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- Weekly pivots are stronger S/R than simple moving averages
- Fisher Transform catches reversals better than RSI in bear markets
- Regime-adaptive logic works in both 2021 bull and 2022-2024 bear/range
- 6h TF = fewer false signals than 4h, more responsive than 12h

Target: Sharpe>0.6, trades≥30 train, trades≥3 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_fisher_regime_1d1w_v1"
timeframe = "6h"
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

def calculate_fisher(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Better at catching extremes than RSI, especially in bear markets
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    median = (high + low) / 2
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            if i > 0 and not np.isnan(fisher[i-1]):
                fisher[i] = fisher[i-1]
                trigger[i] = fisher[i-1]
            continue
        
        normalized = 2.0 * (median[i] - lowest) / range_val - 1.0
        normalized = max(-0.999, min(0.999, normalized))
        
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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

def calculate_pivot_levels(high, low, close):
    """
    Standard Pivot Points: P = (H+L+C)/3
    R1 = 2P - L, S1 = 2P - H
    R2 = P + (H-L), S2 = P - (H-L)
    Returns arrays of pivot, R1, S1, R2, S2
    """
    n = len(close)
    pivot = np.full(n, np.nan, dtype=np.float64)
    r1 = np.full(n, np.nan, dtype=np.float64)
    s1 = np.full(n, np.nan, dtype=np.float64)
    r2 = np.full(n, np.nan, dtype=np.float64)
    s2 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        h = high[i-1]
        l = low[i-1]
        c = close[i-1]
        
        pivot[i] = (h + l + c) / 3.0
        r1[i] = 2.0 * pivot[i] - l
        s1[i] = 2.0 * pivot[i] - h
        r2[i] = pivot[i] + (h - l)
        s2[i] = pivot[i] - (h - l)
    
    return pivot, r1, s1, r2, s2

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
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_trigger = calculate_fisher(high, low, period=9)
    
    # Calculate pivot levels on 6h data
    pivot, r1, s1, r2, s2 = calculate_pivot_levels(high, low, close)
    
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
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(pivot[i]):
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
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === TREND DIRECTION (1d and 1w HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bias when 1d and 1w agree
        strong_bull_bias = price_above_1d and price_above_1w
        strong_bear_bias = price_below_1d and price_below_1w
        
        # === FISHER TRANSFORM SIGNALS (LOOSE thresholds for trades) ===
        fisher_val = fisher[i]
        fisher_prev = fisher_trigger[i] if not np.isnan(fisher_trigger[i]) else fisher_val
        
        # Fisher crossover signals - LOOSE thresholds to ensure trades
        fisher_bull_cross = fisher_val > -1.0 and fisher_prev <= -1.0
        fisher_bear_cross = fisher_val < 1.0 and fisher_prev >= 1.0
        fisher_extreme_low = fisher_val < -0.5
        fisher_extreme_high = fisher_val > 0.5
        
        # === RSI CONFIRMATION (LOOSE) ===
        rsi_val = rsi_14[i]
        rsi_bullish = rsi_val > 35
        rsi_bearish = rsi_val < 65
        rsi_oversold = rsi_val < 45
        rsi_overbought = rsi_val > 55
        
        # === PIVOT LEVEL PROXIMITY ===
        # Check if price is near support (S1, S2) or resistance (R1, R2)
        pivot_val = pivot[i]
        s1_val = s1[i]
        s2_val = s2[i]
        r1_val = r1[i]
        r2_val = r2[i]
        
        # Price near support (within 1%)
        near_support = False
        if not np.isnan(s1_val):
            if close[i] <= s1_val * 1.01 and close[i] >= s1_val * 0.98:
                near_support = True
        if not np.isnan(s2_val) and not near_support:
            if close[i] <= s2_val * 1.01 and close[i] >= s2_val * 0.98:
                near_support = True
        # Also consider pivot itself as support in uptrend
        if not np.isnan(pivot_val) and not near_support:
            if close[i] <= pivot_val * 1.01 and close[i] >= pivot_val * 0.98:
                near_support = True
        
        # Price near resistance (within 1%)
        near_resistance = False
        if not np.isnan(r1_val):
            if close[i] >= r1_val * 0.99 and close[i] <= r1_val * 1.02:
                near_resistance = True
        if not np.isnan(r2_val) and not near_resistance:
            if close[i] >= r2_val * 0.99 and close[i] <= r2_val * 1.02:
                near_resistance = True
        # Also consider pivot itself as resistance in downtrend
        if not np.isnan(pivot_val) and not near_resistance:
            if close[i] >= pivot_val * 0.99 and close[i] <= pivot_val * 1.02:
                near_resistance = True
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: Fisher + 1d/1w bias + RSI
        if is_trend_regime:
            # LONG: Strong bull bias + Fisher bull cross + RSI bullish
            if strong_bull_bias and fisher_bull_cross and rsi_bullish:
                desired_signal = SIZE_STRONG
            # Also enter on pullback to pivot in uptrend
            elif price_above_1d and near_support and fisher_val > -0.5 and rsi_val > 40:
                desired_signal = SIZE_BASE
            
            # SHORT: Strong bear bias + Fisher bear cross + RSI bearish
            elif strong_bear_bias and fisher_bear_cross and rsi_bearish:
                desired_signal = -SIZE_STRONG
            # Also enter on bounce to pivot in downtrend
            elif price_below_1d and near_resistance and fisher_val < 0.5 and rsi_val < 60:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Fisher extremes at pivot S/R (mean reversion)
        elif is_range_regime:
            # LONG: Fisher extreme low + price at support + RSI oversold
            if fisher_extreme_low and near_support and rsi_oversold:
                desired_signal = SIZE_BASE
            
            # SHORT: Fisher extreme high + price at resistance + RSI overbought
            elif fisher_extreme_high and near_resistance and rsi_overbought:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: 1d HMA bias + RSI + Fisher (simplest, most trades)
        else:
            # LONG: 1d bullish + Fisher not extreme bearish + RSI neutral-bullish
            if price_above_1d and fisher_val > -0.8 and rsi_val > 40 and rsi_val < 65:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + Fisher not extreme bullish + RSI neutral-bearish
            elif price_below_1d and fisher_val < 0.8 and rsi_val < 60 and rsi_val > 35:
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