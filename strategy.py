#!/usr/bin/env python3
"""
Experiment #1300: 6h Primary + 1d/1w HTF — Fisher Transform Reversals with Trend Bias

Hypothesis: Pure trend-following fails in bear markets (2022 crash, 2025 range). 
Fisher Transform excels at catching reversals in choppy/bear markets while HTF 
trend filters prevent counter-trend trades in strong trends. This combines the 
best of both worlds.

Key innovations vs failed strategies:
1. Fisher Transform (period=9): Catches reversals at extremes (-1.5/+1.5 levels)
2. Dual HTF bias (1w + 1d): Only trade Fisher signals WITH higher timeframe trend
3. Volume confirmation: Filter out low-volume false breakouts (vol > 20-bar avg)
4. 6h timeframe: Natural 30-60 trades/year (between 4h noise and 12h slowness)
5. Discrete sizing (0.0, ±0.25, ±0.30): Minimize fee churn on signal changes

Why this should beat KAMA+ROC (Sharpe=0.447):
- Fisher Transform has 75% win rate on reversals (Connors research)
- Works in bear markets where trend-following gets whipsawed
- 1w+1d filter prevents dangerous counter-trend trades
- Volume filter removes low-liquidity false signals

Entry logic:
- LONG: 1w_HMA bullish + 1d_HMA bullish + Fisher crosses above -1.5 + vol>avg
- SHORT: 1w_HMA bearish + 1d_HMA bearish + Fisher crosses below +1.5 + vol>avg

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_reversal_htf_trend_1d1w_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals at extreme values (-1.5 to +1.5 typical thresholds)
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to -1 to +1 range using period high/low
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate typical price
    typical = (high + low) / 2.0
    
    for i in range(period, n):
        # Find period high and low
        window_high = np.nanmax(high[i-period+1:i+1])
        window_low = np.nanmin(low[i-period+1:i+1])
        
        price_range = window_high - window_low
        if price_range < 1e-10:
            continue
        
        # Normalize to -1 to +1 (with 0.99 dampening to avoid division issues)
        normalized = 0.99 * (2.0 * (typical[i] - window_low) / price_range - 1.0)
        
        # Clamp to avoid log issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Previous value for crossover detection
        if i > period:
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        window = volume[i-period+1:i+1]
        if not np.any(np.isnan(window)):
            vol_sma[i] = np.mean(window)
    
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    # 6h HMA for local trend confirmation
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (1w + 1d) ===
        # Weekly trend: price above/below 1w HMA
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # Daily trend: price above/below 1d HMA
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 6h local trend
        price_above_6h = close[i] > hma_6h[i] if not np.isnan(hma_6h[i]) else False
        price_below_6h = close[i] < hma_6h[i] if not np.isnan(hma_6h[i]) else False
        
        # === VOLUME CONFIRMATION ===
        volume_above_avg = volume[i] > vol_sma_20[i] * 1.0  # At least average volume
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_val = fisher[i]
        fisher_prev_val = fisher_prev[i]
        
        # Fisher crossover detection
        fisher_cross_up = fisher_prev_val < -1.5 and fisher_val >= -1.5
        fisher_cross_down = fisher_prev_val > 1.5 and fisher_val <= 1.5
        
        # Fisher extreme levels (for strong signals)
        fisher_oversold = fisher_val < -2.0
        fisher_overbought = fisher_val > 2.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + 1d bullish + Fisher reversal from oversold + volume
        if price_above_1w and price_above_1d:
            if fisher_cross_up and volume_above_avg:
                if fisher_oversold:
                    desired_signal = SIZE_STRONG  # Strong reversal signal
                else:
                    desired_signal = SIZE_BASE  # Standard reversal signal
        
        # SHORT: 1w bearish + 1d bearish + Fisher reversal from overbought + volume
        elif price_below_1w and price_below_1d:
            if fisher_cross_down and volume_above_avg:
                if fisher_overbought:
                    desired_signal = -SIZE_STRONG  # Strong reversal signal
                else:
                    desired_signal = -SIZE_BASE  # Standard reversal signal
        
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