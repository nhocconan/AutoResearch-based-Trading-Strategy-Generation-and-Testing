#!/usr/bin/env python3
"""
Experiment #880: 6h Primary + 1d/1w HTF — Fisher Transform Reversals + Dual HMA Bias

Hypothesis: 6h timeframe with weekly/daily HTF bias captures multi-day swings while
avoiding noise. Fisher Transform (Ehlers) provides superior reversal detection vs RSI
in bear/range markets (2025+). Dual HMA (1d + 1w) ensures we only trade WITH the
higher-timeframe trend, reducing whipsaw. Volatility ratio (ATR7/ATR30) filters
low-vol traps.

Key innovations:
1. 1w HMA(21) + 1d HMA(21) dual bias - both must agree for high-conviction trades
2. 6h Fisher Transform(9) for entry timing - catches reversals at extremes
3. ATR ratio(7/30) > 1.5 = vol expansion (validates breakout/reversal)
4. Loose Fisher thresholds: < -1.2 long, > +1.2 short (ensures ≥10 trades)
5. 2.5x ATR trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure trades):
- LONG: 1w HMA bull + 1d HMA bull + Fisher < -1.2 + ATR_ratio > 1.3
- SHORT: 1w HMA bear + 1d HMA bear + Fisher > +1.2 + ATR_ratio > 1.3
- If only 1d agrees (1w neutral), use half size (0.20)

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_dual_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    if sqrt_n < 1:
        sqrt_n = 1
    
    # WMA helper
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    # WMA of diff with sqrt(n)
    hma = wma(diff, sqrt_n)
    return hma

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for clearer reversal signals
    Fisher crosses +2 = overbought (short), crosses -2 = oversold (long)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate typical price
    tp = (high + low + close) / 3.0
    
    # Normalize price to -1 to +1 range
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        price_range = highest - lowest
        
        if price_range < 1e-10:
            continue
        
        # Normalize: (2 * (tp - lowest) / range) - 1
        normalized = 2.0 * (tp[i] - lowest) / price_range - 1.0
        
        # Clamp to avoid division by zero in next step
        normalized = np.clip(normalized, -0.99, 0.99)
        
        # Fisher transform: 0.5 * ln((1 + x) / (1 - x))
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Trigger line (previous Fisher)
        if i > 0 and not np.isnan(fisher[i - 1]):
            trigger[i] = fisher[i - 1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA (Rule 2 - use align_htf_to_ltf)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher, trigger = calculate_fisher_transform(high, low, close, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # ATR ratio for vol regime
    atr_ratio = np.full(n, np.nan)
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_WEAK = 0.20
    
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
        
        if np.isnan(fisher[i]) or np.isnan(trigger[i]) or np.isnan(atr_ratio[i]):
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
        
        # === HTF BIAS (1w + 1d HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Dual bias agreement
        dual_bull = htf_1w_bull and htf_1d_bull
        dual_bear = htf_1w_bear and htf_1d_bear
        single_bull = htf_1d_bull and not htf_1w_bull
        single_bear = htf_1d_bear and not htf_1w_bear
        
        # === FISHER TRANSFORM SIGNALS (LOOSE THRESHOLDS FOR TRADES) ===
        fisher_oversold = fisher[i] < -1.2  # Long entry
        fisher_overbought = fisher[i] > 1.2  # Short entry
        
        # Fisher crossover confirmation
        fisher_cross_long = (trigger[i] < -1.2) and (fisher[i] >= trigger[i]) and (fisher[i] < 0)
        fisher_cross_short = (trigger[i] > 1.2) and (fisher[i] <= trigger[i]) and (fisher[i] > 0)
        
        # === VOLATILITY REGIME ===
        vol_expansion = atr_ratio[i] > 1.3  # Vol expanding (validates move)
        
        # === ENTRY LOGIC (LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        if dual_bull:
            # Strong bullish bias - look for long entries
            if fisher_oversold or fisher_cross_long:
                if vol_expansion:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        elif dual_bear:
            # Strong bearish bias - look for short entries
            if fisher_overbought or fisher_cross_short:
                if vol_expansion:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        elif single_bull:
            # Weaker bullish bias (only 1d agrees)
            if fisher_oversold:
                desired_signal = SIZE_WEAK
        elif single_bear:
            # Weaker bearish bias (only 1d agrees)
            if fisher_overbought:
                desired_signal = -SIZE_WEAK
        
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
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
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