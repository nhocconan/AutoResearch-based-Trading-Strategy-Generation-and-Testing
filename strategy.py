#!/usr/bin/env python3
"""
Experiment #1215: 6h Primary + 12h/1d HTF — Fisher Transform + Choppiness Regime + Multi-TF HMA

Hypothesis: Pure trend-following (HMA/EMA crossover) fails in bear/range markets (2022 crash, 2025 test).
The Fisher Transform excels at catching reversals in bear markets by normalizing price to Gaussian distribution.
Combined with Choppiness Index regime filter and 12h/1d HMA trend bias, this should outperform simple trend strategies.

Key innovations vs failed 6h strategies:
1. Fisher Transform (period=9) for entry timing - catches reversals at extremes (-1.5/+1.5)
2. Choppiness Index (period=14) regime filter - only trade when CHOP confirms regime
3. Volume confirmation - entry volume > 20-bar MA volume (filters low-conviction moves)
4. ATR ratio filter - skip entries when ATR(7)/ATR(30) > 2.0 (volatility spike = wait)
5. Multi-TF HMA alignment - 1d HMA for primary bias, 12h HMA for intermediate confirmation

Entry logic:
- LONG: Fisher crosses above -1.5 + price > 1d_HMA + CHOP < 55 (trending) OR CHOP > 55 (range bounce)
- SHORT: Fisher crosses below +1.5 + price < 1d_HMA + volume confirmation
- Size: 0.25 base, 0.30 strong (all HTF aligned)
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_regime_hma_12h1d_v1"
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
    Catches reversals at extreme values (-1.5 to +1.5 typical bounds)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    # Normalize to 0-1 range using highest high and lowest low over period
    fisher_input = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        if highest > lowest:
            fisher_input[i] = (typical[i] - lowest) / (highest - lowest)
    
    # Clamp to 0.001-0.999 to avoid log(0)
    fisher_input = np.clip(fisher_input, 0.001, 0.999)
    
    # Fisher transform
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        if not np.isnan(fisher_input[i]) and not np.isnan(fisher_input[i-1]):
            fisher[i] = 0.5 * np.log((1 + fisher_input[i]) / (1 - fisher_input[i]))
            fisher_prev[i] = 0.5 * np.log((1 + fisher_input[i-1]) / (1 - fisher_input[i-1]))
    
    return fisher, fisher_prev

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - identifies trending vs ranging markets
    CHOP > 61.8 = range (mean reversion favorable)
    CHOP < 38.2 = trend (trend following favorable)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        if highest > lowest:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def calculate_volume_ma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        vol_ma[i] = np.nanmean(volume[i-period+1:i+1])
    
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness_index(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    
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
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (Daily + 12h HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        # === REGIME FILTER (Choppiness Index) ===
        chop_value = chop[i]
        is_trending = chop_value < 55.0  # Below 55 = trending regime
        is_ranging = chop_value >= 55.0  # Above 55 = ranging regime
        
        # === VOLATILITY FILTER (ATR ratio) ===
        atr_ratio = atr_7[i] / atr_30[i] if atr_30[i] > 1e-10 else 999.0
        vol_spike = atr_ratio > 2.0  # Skip entries during vol spikes
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = not np.isnan(vol_ma[i]) and volume[i] > vol_ma[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_cross_down = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Fisher crosses above -1.5 + bullish trend alignment
        if fisher_cross_up and not vol_spike:
            if price_above_1d and price_above_12h:
                # Strong long: all HTF aligned + volume confirmation
                if vol_confirmed or is_trending:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            elif price_above_1d:
                # Basic long: only 1d aligned
                if vol_confirmed:
                    desired_signal = SIZE_BASE
        
        # SHORT: Fisher crosses below +1.5 + bearish trend alignment
        elif fisher_cross_down and not vol_spike:
            if price_below_1d and price_below_12h:
                # Strong short: all HTF aligned + volume confirmation
                if vol_confirmed or is_trending:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            elif price_below_1d:
                # Basic short: only 1d aligned
                if vol_confirmed:
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