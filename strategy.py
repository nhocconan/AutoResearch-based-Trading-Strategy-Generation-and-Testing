#!/usr/bin/env python3
"""
Experiment #1243: 6h Primary + 1d/1w HTF — Ehlers Fisher Transform + Volatility Regime

Hypothesis: After 960+ failed experiments, trend-following alone fails in bear/range markets.
The Ehlers Fisher Transform (1990s) is specifically designed to catch reversals by transforming
price into a Gaussian distribution, making turning points more visible. Combined with HTF trend
bias and volatility regime filter, this should work better than pure trend or pure mean-reversion.

Key innovations vs failed 6h strategies:
1. Fisher Transform instead of RSI - better at identifying true turning points
2. Volatility compression filter (ATR ratio) - only enter when vol is normalizing
3. 1d HMA for trend bias, 1w HMA for major trend confirmation
4. LOOSE entry thresholds to guarantee trades (Fisher cross is frequent)
5. ATR trailing stop for risk management

Entry logic:
- LONG: Fisher crosses above Trigger AND price > 1d_HMA AND ATR_ratio < 1.5
- SHORT: Fisher crosses below Trigger AND price < 1d_HMA AND ATR_ratio < 1.5
- Strong signal: also aligned with 1w HMA

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_transform_vol_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - transforms price into Gaussian distribution
    Makes turning points more visible for reversal trading
    Reference: Ehlers, J.F. (2002) "Fisher Transform"
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    # Normalize price to range -1 to +1
    fisher_input = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        highest = np.nanmax(typical[i - period + 1:i + 1])
        lowest = np.nanmin(typical[i - period + 1:i + 1])
        if highest > lowest:
            fisher_input[i] = 0.66 * ((typical[i] - lowest) / (highest - lowest) - 0.5) + 0.67 * fisher_input[i - 1]
            fisher_input[i] = np.clip(fisher_input[i], -0.999, 0.999)
    
    # Fisher transform
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(fisher_input[i]) and abs(fisher_input[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1.0 + fisher_input[i]) / (1.0 - fisher_input[i]))
            if not np.isnan(fisher[i - 1]):
                trigger[i] = fisher[i - 1]
    
    return fisher, trigger

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

def calculate_atr_ratio(atr, short_period=7, long_period=30):
    """ATR ratio to detect volatility spikes"""
    n = len(atr)
    if n < long_period:
        return np.full(n, np.nan)
    
    # Use rolling mean as proxy for ATR smoothing
    atr_short = pd.Series(atr).rolling(window=short_period, min_periods=short_period).mean().values
    atr_long = pd.Series(atr).rolling(window=long_period, min_periods=long_period).mean().values
    
    ratio = np.divide(atr_short, atr_long, out=np.zeros_like(atr_short), where=atr_long > 0)
    return ratio

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
    atr_ratio = calculate_atr_ratio(atr_14, short_period=7, long_period=30)
    fisher, trigger = calculate_fisher_transform(high, low, close, period=9)
    
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
        
        if np.isnan(fisher[i]) or np.isnan(trigger[i]):
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
        
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (Daily HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # Weekly HMA for additional confirmation
        hma_1w_valid = not np.isnan(hma_1w_aligned[i])
        price_above_1w = hma_1w_valid and close[i] > hma_1w_aligned[i]
        price_below_1w = hma_1w_valid and close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY REGIME FILTER ===
        # Only enter when volatility is normalizing (not spiking)
        vol_normal = atr_ratio[i] < 1.5
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = False
        fisher_cross_down = False
        
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(trigger[i-1]):
            # Fisher crosses above trigger (bullish reversal)
            if fisher[i-1] <= trigger[i-1] and fisher[i] > trigger[i]:
                fisher_cross_up = True
            # Fisher crosses below trigger (bearish reversal)
            if fisher[i-1] >= trigger[i-1] and fisher[i] < trigger[i]:
                fisher_cross_down = True
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: Fisher cross up + price above 1d HMA + vol normal
        if price_above_1d and fisher_cross_up and vol_normal:
            if price_above_1w:
                desired_signal = SIZE_STRONG  # Strong trend alignment
            else:
                desired_signal = SIZE_BASE  # Basic uptrend reversal
        
        # SHORT: Fisher cross down + price below 1d HMA + vol normal
        elif price_below_1d and fisher_cross_down and vol_normal:
            if price_below_1w:
                desired_signal = -SIZE_STRONG  # Strong trend alignment
            else:
                desired_signal = -SIZE_BASE  # Basic downtrend reversal
        
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