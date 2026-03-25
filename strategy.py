#!/usr/bin/env python3
"""
Experiment #1123: 6h Primary + 1d/1w HTF — Fisher Transform Reversals + HMA Trend Bias

Hypothesis: Ehlers Fisher Transform excels at catching turning points in ranging markets
(which BTC/ETH spend 60-70% of time in). Combined with 1d/1w HMA trend bias for
directional filtering, this should outperform pure trend-following or pure mean-reversion.

Key innovations:
1. Fisher Transform (period=9): Normalizes price to Gaussian distribution, extreme values
   (-2 to +2) mark reversal zones. More responsive than RSI for turning points.
2. 1d HMA(21) + 1w HMA(21): Dual HTF trend bias - only take longs above 1w_HMA, shorts below
3. ATR Ratio Filter (ATR7/ATR30): Avoid entries during low-volatility compression (<0.8)
4. Asymmetric thresholds: Easier entries when aligned with HTF trend, harder against
5. 2.5x ATR trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why 6h timeframe:
- Captures multi-day swings (3-7 day holds typical)
- Less noise than 4h, more trades than 12h
- Target: 30-60 trades/year (1-2 per week per symbol)

Entry conditions (LOOSE to guarantee trades):
- LONG: Fisher < -1.0 + price > 1w_HMA*0.98 (bull bias) OR Fisher < -1.8 (deep oversold any regime)
- SHORT: Fisher > 1.0 + price < 1w_HMA*1.02 (bear bias) OR Fisher > 1.8 (deep overbought any regime)
- ATR ratio > 0.7 to ensure sufficient volatility

Why this should work:
- Fisher Transform has proven edge in ranging markets (BTC 2022-2023, 2025 bear)
- HTF bias prevents counter-trend trades that get stopped out
- ATR filter avoids dead zones with no follow-through
- 6h captures swing moves without 4h noise or 12h slowness

Target: Sharpe>0.50, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_atr_regime_1d1w_v1"
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
    Extreme values (-2 to +2) mark reversal zones
    
    Formula:
    1. Calculate typical price: (High + Low + Close) / 3
    2. Normalize: 0.66 * ((TP - LL) / (HH - LL) - 0.5) + 0.66 * prev_norm
    3. Fisher: 0.5 * ln((1 + norm) / (1 - norm)) + 0.5 * prev_fisher
    
    Bounds normalized value to [-0.99, 0.99] to avoid ln(0)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    tp = (high + low + close) / 3.0
    
    # Normalize and Fisher
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    norm = np.zeros(n, dtype=np.float64)
    
    for i in range(period - 1, n):
        # Find highest high and lowest low over lookback
        hh = np.nanmax(high[i-period+1:i+1])
        ll = np.nanmin(low[i-period+1:i+1])
        
        price_range = hh - ll
        if price_range < 1e-10:
            continue
        
        # Normalize price position within range
        norm_raw = 0.66 * ((tp[i] - ll) / price_range - 0.5)
        
        # Add inertia from previous normalized value
        if i > period - 1 and not np.isnan(norm[i-1]):
            norm[i] = norm_raw + 0.66 * norm[i-1]
        else:
            norm[i] = norm_raw
        
        # Bound to avoid ln(0)
        norm_bounded = np.clip(norm[i], -0.99, 0.99)
        
        # Fisher transform
        fisher_raw = 0.5 * np.log((1 + norm_bounded) / (1 - norm_bounded))
        
        # Add inertia from previous Fisher value
        if i > period - 1 and not np.isnan(fisher[i-1]):
            fisher[i] = fisher_raw + 0.5 * fisher[i-1]
        else:
            fisher[i] = fisher_raw
        
        # Fisher signal is previous Fisher value (for crossover detection)
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

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
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
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
        
        # === VOLATILITY FILTER (ATR Ratio) ===
        # Avoid low volatility periods where moves lack follow-through
        atr_ratio = atr_7[i] / atr_30[i] if atr_30[i] > 1e-10 else 0.0
        vol_ok = atr_ratio > 0.7  # Sufficient volatility
        
        # === HTF TREND BIAS ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong trend alignment (both 1d and 1w agree)
        strong_bull = hma_1d_bull and hma_1w_bull
        strong_bear = hma_1d_bear and hma_1w_bear
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher extreme values mark reversal zones
        fisher_oversold = fisher[i] < -1.0
        fisher_deep_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.0
        fisher_deep_overbought = fisher[i] > 1.8
        
        # Fisher crossover (turning up from oversold / turning down from overbought)
        fisher_turning_up = fisher[i] > fisher_signal[i] and fisher_signal[i] < -0.5
        fisher_turning_down = fisher[i] < fisher_signal[i] and fisher_signal[i] > 0.5
        
        # === ENTRY LOGIC (ASYMMETRIC BASED ON HTF BIAS) ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        if vol_ok:
            # Strong long: Fisher deep oversold (any regime) - high conviction reversal
            if fisher_deep_oversold:
                desired_signal = SIZE_STRONG
            # Moderate long: Fisher oversold + HTF bull bias
            elif fisher_oversold and hma_1w_bull:
                desired_signal = SIZE_BASE
            # Moderate long: Fisher turning up from oversold + HTF bull
            elif fisher_turning_up and strong_bull:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRIES
        if vol_ok:
            # Strong short: Fisher deep overbought (any regime) - high conviction reversal
            if fisher_deep_overbought:
                desired_signal = -SIZE_STRONG
            # Moderate short: Fisher overbought + HTF bear bias
            elif fisher_overbought and hma_1w_bear:
                desired_signal = -SIZE_BASE
            # Moderate short: Fisher turning down from overbought + HTF bear
            elif fisher_turning_down and strong_bear:
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