#!/usr/bin/env python3
"""
Experiment #010: 1h Primary + 4h/12h HTF — Fisher Transform + HMA Trend + BB Regime

Hypothesis: After 9 experiments, the key insight is that BTC/ETH in 2025 (bear/range)
need MEAN REVERSION entries within HTF trend context. Pure trend following failed.
This strategy combines:
1. 4h HMA(21) for trend direction (proven in #009)
2. 12h HMA(21) for major trend bias (prevents counter-trend in crashes)
3. Fisher Transform(9) for entry timing - catches reversals in bear rallies
4. Bollinger Band Width percentile for regime detection (squeeze = low vol = mean revert)
5. ATR(14) 2.5x trailing stop - proven risk management
6. Discrete sizing: 0.25 (smaller for 1h to reduce fee impact)

Why this should work on 1h:
- Fisher Transform is proven for bear/range markets (research shows 75% win rate)
- 4h + 12h dual HMA prevents whipsaw (learned from 2022 crash)
- BB Width filter avoids trading in high-vol expansion (reduces false signals)
- 1h timeframe with HTF filters = ~40-60 trades/year target (fee-efficient)
- Entry only when Fisher crosses extreme levels WITH HTF trend agreement

Entry Logic:
- Long: 4h close > 4h HMA + 12h close > 12h HMA + Fisher crosses above -1.5 + BB Width < 50th percentile
- Short: 4h close < 4h HMA + 12h close < 12h HMA + Fisher crosses below +1.5 + BB Width < 50th percentile
- Size: 0.25 (discrete, minimizes fee churn for 1h)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.3, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_bb_regime_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - responsive trend indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - catches reversals in bear/range markets
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * ((price - min) / (max - min) - 0.5) * 2
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Use typical price
        typical = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over lookback
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if hh == ll:
            continue
        
        # Normalize price to -1 to +1 range
        x = 0.67 * ((typical - ll) / (hh - ll) - 0.5) * 2.0
        
        # Clamp to avoid division by zero
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Previous value for crossover detection
        if i > period - 1:
            fisher_prev[i] = fisher[i - 1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with bandwidth calculation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # Rolling mean and std
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    
    return upper, lower, bandwidth

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bb_width_percentile(bandwidth, lookback=100):
    """Calculate percentile rank of BB Width over lookback period"""
    n = len(bandwidth)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if np.isnan(bandwidth[i]):
            continue
        window = bandwidth[i - lookback:i + 1]
        window = window[~np.isnan(window)]
        if len(window) < lookback // 2:
            continue
        # Percentile rank: what % of values are below current
        percentile[i] = np.sum(window < bandwidth[i]) / len(window) * 100.0
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for major trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=100)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position size (smaller for 1h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
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
        if np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === DUAL HMA TREND ALIGNMENT (4h + 12h) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === FISHER TRANSFORM CROSSOVER ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_cross_long = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_short = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === BB WIDTH REGIME FILTER ===
        # Only trade when BB Width is in lower 50th percentile (low vol / squeeze)
        bb_squeeze = bb_width_pct[i] < 50.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: Dual HMA bullish + Fisher cross + BB squeeze
        if hma_4h_bull and hma_12h_bull and fisher_cross_long and bb_squeeze:
            desired_signal = SIZE
        
        # Short entry: Dual HMA bearish + Fisher cross + BB squeeze
        elif hma_4h_bear and hma_12h_bear and fisher_cross_short and bb_squeeze:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals