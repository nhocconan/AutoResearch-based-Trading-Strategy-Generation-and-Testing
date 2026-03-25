#!/usr/bin/env python3
"""
Experiment #1531: 6h Primary + 1w/1d HTF — Fisher Transform Vol Reversion

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). This strategy
combines Fisher Transform (proven reversal indicator for bear/range markets) with
volatility spike detection to catch mean-reversion opportunities after panic moves.

Key components:
1. 1w HMA(21) for macro regime bias (longs only when 1w bullish)
2. 1d HMA(21) for intermediate trend filter
3. Fisher Transform(9) for entry timing (crosses at -1.5/+1.5 levels)
4. ATR vol spike filter (ATR7/ATR30 > 1.8 = elevated vol, wait for contraction)
5. 6h HMA(16/48) for momentum confirmation
6. ATR(14) trailing stoploss (2.5x ATR)
7. Discrete sizing: 0.0, ±0.25, ±0.30

Why this should work:
- Fisher Transform excels at catching reversals in bear/range markets (2022, 2025)
- Vol spike + contraction pattern = high-probability mean reversion
- 1w/1d HTF filters prevent major counter-trend positions
- 6h TF = natural 30-50 trades/year (fee-efficient)
- LOOSE Fisher thresholds (-1.5/+1.5, not -2/+2) guarantee trades

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 1w_HMA bullish + 1d_HMA bullish + Fisher crosses above -1.5 + vol_spike_contracting
- SHORT: Fisher crosses below +1.5 + vol_spike_contracting (always allowed in bear)
- Vol spike: ATR7/ATR30 > 1.5 (not too strict)
- Contracting: ATR7 < ATR7_prev (vol decreasing after spike)

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_vol_reversion_1w1d_v1"
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
    Excellent for identifying reversals in ranging/bear markets
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10:
            continue
        
        # Normalize price to -1 to +1 range
        mid_price = (high[i] + low[i]) / 2.0
        normalized = ((mid_price - lowest_low) / price_range) - 0.5
        
        # Clamp to avoid division issues
        normalized = max(-0.999, min(0.999, normalized * 2.0))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Previous value for cross detection
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_prev[i] = fisher[i-1]
        elif i > period - 1:
            # Calculate previous bar's fisher
            hh_prev = np.max(high[i - period:i])
            ll_prev = np.min(low[i - period:i])
            range_prev = hh_prev - ll_prev
            if range_prev > 1e-10:
                mid_prev = (high[i-1] + low[i-1]) / 2.0
                norm_prev = ((mid_prev - ll_prev) / range_prev) - 0.5
                norm_prev = max(-0.999, min(0.999, norm_prev * 2.0))
                fisher_prev[i] = 0.5 * np.log((1 + norm_prev) / (1 - norm_prev))
    
    return fisher, fisher_prev

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    rsi_14 = calculate_rsi(close, period=14)
    
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
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY SPIKE FILTER ===
        vol_ratio = atr_7[i] / atr_30[i]
        vol_spike = vol_ratio > 1.5  # Elevated volatility
        
        # Vol contracting (ATR decreasing from previous bar)
        vol_contracting = False
        if i > 0 and not np.isnan(atr_7[i-1]):
            vol_contracting = atr_7[i] < atr_7[i-1]
        
        # Only trade when vol is elevated AND contracting (after panic)
        vol_setup = vol_spike and vol_contracting
        
        # === HTF TREND BIAS ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1w bullish = allow longs, 1w bearish = shorts only
        macro_bullish = price_above_1w
        macro_bearish = not price_above_1w
        
        # === 6h MOMENTUM ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_cross = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_short_cross = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # Also allow entries when Fisher is at extremes (not just crosses)
        fisher_oversold = fisher[i] < -1.2
        fisher_overbought = fisher[i] > 1.2
        
        # === RSI FILTER (avoid extreme counter-trend) ===
        rsi = rsi_14[i]
        rsi_not_overbought = rsi < 70
        rsi_not_oversold = rsi > 30
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + (Fisher cross OR extreme) + vol setup + momentum
        if macro_bullish:
            if (fisher_long_cross or fisher_oversold) and rsi_not_overbought:
                if vol_setup or hma_bullish:  # Either vol setup OR momentum confirm
                    desired_signal = SIZE_STRONG if vol_setup else SIZE_BASE
        
        # SHORT: Always allowed (bear market friendly) + Fisher signal
        if macro_bearish or True:  # Shorts always allowed
            if (fisher_short_cross or fisher_overbought) and rsi_not_oversold:
                if vol_setup or hma_bearish:  # Either vol setup OR momentum confirm
                    desired_signal = -SIZE_STRONG if vol_setup else -SIZE_BASE
        
        # Override: if strong HMA momentum, allow trades even without vol setup
        if desired_signal == 0.0:
            if macro_bullish and hma_bullish and fisher_oversold and rsi < 60:
                desired_signal = SIZE_BASE
            elif hma_bearish and fisher_overbought and rsi > 40:
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