#!/usr/bin/env python3
"""
Experiment #1657: 1d Primary + 1w HTF — Fisher Transform Reversals with Trend Filter

Hypothesis: Recent dual-regime strategies (#1653-1656) either got 0 trades or negative Sharpe.
The Fisher Transform excels at catching reversals in bear/range markets (2025 test period).
Combined with 1w HMA trend filter, this should generate 25-40 high-quality trades/year.

Why this should work:
1. Fisher Transform normalizes price to Gaussian distribution, better at extremes than RSI
2. 1w HMA provides trend bias without over-filtering (unlike dual-regime CHOP+CRSI+HMA)
3. Asymmetric sizing: 0.30 with trend, 0.15 counter-trend (reduces whipsaw damage)
4. Simple entry: Fisher cross + price position vs HMA
5. ATR 2.5x trailing stop (tighter than 3.0x to protect gains in bear market)

Key differences from failed experiments:
- Fewer conditions (Fisher + 1w HMA only, no CHOP/CRSI/Donchian combo)
- Looser Fisher thresholds (-1.5/+1.5 not -2.0/+2.0)
- Single HTF (1w) not multi-HTF (reduces complexity)

Target: Sharpe > 0.618, trades > 30/symbol train, > 5/symbol test, DD > -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_1w_hma_atr_asym_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Transforms price into a Gaussian normal distribution
    Long when Fisher crosses above -1.5 (oversold reversal)
    Short when Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(close)
    if n < period + 10:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    # Smooth with EMA
    ema_typical = pd.Series(typical).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Normalize to -1 to +1 range
    normalized = np.full(n, np.nan)
    for i in range(period, n):
        highest = np.max(ema_typical[i - period + 1:i + 1])
        lowest = np.min(ema_typical[i - period + 1:i + 1])
        
        if highest == lowest:
            normalized[i] = 0.0
        else:
            normalized[i] = 0.66 * ((ema_typical[i] - lowest) / (highest - lowest) - 0.5) + 0.67 * normalized[i-1] if i > period else 0.0
            # Clamp to -0.99 to +0.99
            normalized[i] = np.clip(normalized[i], -0.99, 0.99)
    
    # Fisher transform
    for i in range(period, n):
        if np.isnan(normalized[i]) or abs(normalized[i]) >= 0.999:
            continue
        fisher[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]))
        if i > period and not np.isnan(fisher[i-1]):
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

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

def calculate_rsi(close, period=14):
    """Relative Strength Index - additional filter"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if loss_smooth[i-1] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + gain_smooth[i-1] / loss_smooth[i-1]))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1w HMA direction (slope)
    hma_1w_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i-1]):
            hma_1w_slope[i] = hma_1w_aligned[i] - hma_1w_aligned[i-1]
    
    # Calculate primary (1d) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing
    SIZE_WITH_TREND = 0.30
    SIZE_COUNTER_TREND = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1w_slope[i]):
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
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w TREND BIAS ===
        weekly_bullish = hma_1w_slope[i] > 0
        weekly_bearish = hma_1w_slope[i] < 0
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === RSI FILTER (avoid extreme counter-trend entries) ===
        rsi_ok_long = rsi[i] < 70  # Don't long when already overbought
        rsi_ok_short = rsi[i] > 30  # Don't short when already oversold
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if fisher_long and rsi_ok_long:
            # Long signal
            if weekly_bullish:
                desired_signal = SIZE_WITH_TREND
            else:
                desired_signal = SIZE_COUNTER_TREND
        
        elif fisher_short and rsi_ok_short:
            # Short signal
            if weekly_bearish:
                desired_signal = -SIZE_WITH_TREND
            else:
                desired_signal = -SIZE_COUNTER_TREND
        
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
        if desired_signal >= SIZE_WITH_TREND * 0.85:
            final_signal = SIZE_WITH_TREND
        elif desired_signal <= -SIZE_WITH_TREND * 0.85:
            final_signal = -SIZE_WITH_TREND
        elif desired_signal >= SIZE_COUNTER_TREND * 0.85:
            final_signal = SIZE_COUNTER_TREND
        elif desired_signal <= -SIZE_COUNTER_TREND * 0.85:
            final_signal = -SIZE_COUNTER_TREND
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