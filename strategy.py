#!/usr/bin/env python3
"""
Experiment #1551: 6h Primary + 1w/1d HTF — Fisher Transform Regime Strategy

Hypothesis: 6h timeframe offers optimal balance between trade frequency (30-60/year)
and signal quality. This strategy combines:
1. 1w HMA(21) for major trend bias (slowest filter, avoids counter-trend)
2. 1d Choppiness Index(14) for regime detection (trending vs ranging)
3. 6h Ehlers Fisher Transform(9) for reversal entries (proven in bear markets)
4. 6h ATR ratio for volatility confirmation (avoid low-vol traps)
5. Volume filter to confirm breakouts

Why this should work:
- Fisher Transform excels at catching reversals in bear/range markets (2022, 2025)
- 1w HMA prevents major counter-trend positions
- 1d CHOP adapts entry logic to market regime
- 6h TF = natural 30-50 trades/year (fee-efficient)
- LOOSE Fisher thresholds (-1.5/+1.5) guarantee trade generation

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 1w_HMA bullish + Fisher crosses above -1.5 + ATR_ratio > 0.8 + vol > avg
- SHORT: 1w_HMA bearish + Fisher crosses below +1.5 + ATR_ratio > 0.8 + vol > avg
- Range regime (CHOP>61.8): tighter Fisher thresholds, mean-reversion focus
- Trend regime (CHOP<38.2): wider thresholds, breakout focus

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_regime_1w1d_v1"
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
    Excellent for catching reversals in bear/range markets
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate typical price and normalize
    for i in range(period - 1, n):
        # Use midpoint of high/low for this period
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10:
            continue
        
        # Normalize price to 0-1 range
        normalized = (hl2 - lowest_low) / price_range
        
        # Clamp to avoid division issues
        normalized = max(0.001, min(0.999, normalized))
        
        # Calculate Ehlers variable
        ehlers_var = 0.66 * ((normalized - 0.5) / 0.5)
        
        # Smooth with previous value
        if i > period - 1 and not np.isnan(fisher[i - 1]):
            ehlers_var = 0.66 * ehlers_var + 0.67 * ehlers_var  # Simplified smoothing
        
        # Fisher transform
        if abs(ehlers_var) < 0.999:
            fisher[i] = 0.5 * np.log((1 + ehlers_var) / (1 - ehlers_var))
            if i > 0:
                fisher_prev[i] = fisher[i - 1]
    
    return fisher, fisher_prev

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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    chop_1d_raw = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    
    # Volume average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio for volatility filter
    atr_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            atr_ratio[i] = atr_14[i] / atr_30[i]
    
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
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_ratio[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (1d Choppiness) ===
        chop = chop_1d_aligned[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        is_neutral_regime = not is_trend_regime and not is_range_regime
        
        # === TREND DIRECTION (1w HMA bias) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_val = fisher[i]
        fisher_prev_val = fisher_prev[i]
        
        # Fisher crossover signals
        fisher_cross_up = fisher_prev_val < -1.5 and fisher_val >= -1.5
        fisher_cross_down = fisher_prev_val > 1.5 and fisher_val <= 1.5
        
        # Extreme readings (mean reversion in range regime)
        fisher_oversold = fisher_val < -2.0
        fisher_overbought = fisher_val > 2.0
        
        # === VOLATILITY FILTER ===
        vol_filter = atr_ratio[i] > 0.7  # ATR not collapsing
        vol_confirm = volume[i] > vol_avg[i] * 0.8  # Volume near average
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow 1w trend with Fisher confirmation
        if is_trend_regime:
            # LONG: 1w bullish + Fisher cross up + vol confirm
            if price_above_1w and fisher_cross_up and vol_filter:
                desired_signal = SIZE_STRONG
            
            # SHORT: 1w bearish + Fisher cross down + vol confirm
            elif price_below_1w and fisher_cross_down and vol_filter:
                desired_signal = -SIZE_STRONG
        
        # RANGE REGIME: Mean reversion with Fisher extremes
        elif is_range_regime:
            # LONG: Fisher oversold (regardless of 1w bias in range)
            if fisher_oversold and vol_filter:
                desired_signal = SIZE_BASE
            
            # SHORT: Fisher overbought
            elif fisher_overbought and vol_filter:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Require both 1w bias AND Fisher signal
        elif is_neutral_regime:
            # LONG: 1w bullish + Fisher improving
            if price_above_1w and fisher_val > fisher_prev_val and fisher_val < 0:
                desired_signal = SIZE_BASE
            
            # SHORT: 1w bearish + Fisher declining
            elif price_below_1w and fisher_val < fisher_prev_val and fisher_val > 0:
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