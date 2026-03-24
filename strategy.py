#!/usr/bin/env python3
"""
Experiment #1055: 6h Primary + 12h/1d HTF — Fisher Transform + Choppiness Regime + HMA Bias

Hypothesis: 6h timeframe is underexplored middle-ground between 4h swing and 12h position trading.
Using Ehlers Fisher Transform for precise reversal entries, combined with Choppiness regime filter
and 12h/1d HMA trend bias, should capture multi-day swings with fewer whipsaws than 4h.

Key innovations:
1. Ehlers Fisher Transform (period=9): Normalizes price to Gaussian distribution, extreme values
   indicate reversals. Long when Fisher crosses above -1.5, short when crosses below +1.5
2. Choppiness Index (CHOP 14): Regime filter - avoid trend entries in choppy markets
3. 12h HMA(21) + 1d HMA(21): Dual HTF trend bias - only long if both aligned bullish
4. LOOSE entry conditions: Fisher thresholds widened to ensure 30-60 trades/year
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why 6h should work:
- Captures 2-4 day swings (longer than 4h, shorter than 12h)
- Less noise than 4h, more signals than 12h
- Fisher Transform excels at catching reversals in bear/range markets (2022-2025)
- Dual HTF (12h+1d) provides stronger trend confirmation than single HTF

Entry conditions (LOOSE to guarantee trades):
- LONG: Fisher crosses above -1.5 OR Fisher < -1.0 + price > 12h_HMA + CHOP < 65
- SHORT: Fisher crosses below +1.5 OR Fisher > +1.0 + price < 12h_HMA + CHOP < 65
- CHOP > 65: Only mean-reversion entries (Fisher extremes <-2 or >+2)

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_hma_12h1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Extreme values indicate potential reversals
    
    Steps:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to -1 to +1 range using highest high / lowest low
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    4. Smooth with EMA
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize to -1 to +1
    fisher_raw = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            continue
        
        # Normalize typical price to -0.99 to +0.99
        x = 0.99 * (2.0 * (typical[i] - lowest) / range_val - 1.0)
        x = np.clip(x, -0.99, 0.99)
        
        # Fisher transform
        fisher_raw[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
    
    # Smooth fisher with EMA
    fisher_series = pd.Series(fisher_raw)
    fisher = fisher_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Trigger line (1-period lag)
    trigger[1:] = fisher[:-1]
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
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
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    
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
    
    # Track previous fisher for crossover detection
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 60.0  # Range market
        is_trending = chop_14[i] < 50.0  # Trend market
        
        # === HTF BIAS (12h + 1d HMA alignment) ===
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # Strong trend alignment
        strong_bull = price_above_12h and price_above_1d
        strong_bear = price_below_12h and price_below_1d
        
        # Weak/neutral bias
        neutral = not strong_bull and not strong_bear
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > -1.5 and prev_fisher <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and prev_fisher >= 1.5
        fisher_extreme_low = fisher[i] < -2.0
        fisher_extreme_high = fisher[i] > 2.0
        fisher_moderate_low = fisher[i] < -1.0
        fisher_moderate_high = fisher[i] > 1.0
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE + LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE - use Fisher extremes
            # Long when Fisher extremely oversold
            if fisher_extreme_low:
                desired_signal = SIZE_BASE
            # Short when Fisher extremely overbought
            elif fisher_extreme_high:
                desired_signal = -SIZE_BASE
            # Moderate Fisher + HTF bias
            elif fisher_moderate_low and price_above_12h:
                desired_signal = SIZE_BASE
            elif fisher_moderate_high and price_below_12h:
                desired_signal = -SIZE_BASE
        
        elif is_trending:
            # TREND MODE - use Fisher crossovers + HTF confirmation
            # Long on Fisher cross up + bullish HTF
            if fisher_cross_up and strong_bull:
                desired_signal = SIZE_STRONG
            elif fisher_cross_up and price_above_12h:
                desired_signal = SIZE_BASE
            # Short on Fisher cross down + bearish HTF
            elif fisher_cross_down and strong_bear:
                desired_signal = -SIZE_STRONG
            elif fisher_cross_down and price_below_12h:
                desired_signal = -SIZE_BASE
            # Fallback: simple HTF + RSI
            elif strong_bull and rsi_14[i] > 40.0 and rsi_14[i] < 80.0:
                desired_signal = SIZE_BASE
            elif strong_bear and rsi_14[i] < 60.0 and rsi_14[i] > 20.0:
                desired_signal = -SIZE_BASE
        
        else:
            # NEUTRAL REGIME - loose entries
            if fisher_moderate_low and price_above_12h:
                desired_signal = SIZE_BASE
            elif fisher_moderate_high and price_below_12h:
                desired_signal = -SIZE_BASE
            elif fisher_cross_up:
                desired_signal = SIZE_BASE
            elif fisher_cross_down:
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
        prev_fisher = fisher[i]
    
    return signals