#!/usr/bin/env python3
"""
Experiment #1640: 1h Primary + 4h/12h HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: Previous 1h/30m strategies failed due to OVER-FILTERING (0 trades = Sharpe=0).
This strategy uses PROVEN mean-reversion signals that work in bear/range markets (2022, 2025)
while still capturing trend moves in bull markets (2021).

Key components:
1. HTF bias: 4h HMA(21) for trend direction (simpler than dual HTF = MORE TRADES)
2. Regime filter: Choppiness Index(14) - CHOP>55 = range (mean revert), CHOP<45 = trend (follow)
3. Entry signal: Ehlers Fisher Transform(9) - crosses -1.5 (long) or +1.5 (short)
4. Vol confirmation: ATR(7)/ATR(30) ratio > 1.5 = vol spike (panic/reversal opportunity)
5. Session filter: Only trade 8-20 UTC (high liquidity, avoid Asian chop)
6. ATR(14) 2.5x trailing stop: Controlled drawdown

Why this should work on 1h:
- Fisher Transform catches reversals in bear rallies (proven in 2022 crash)
- Choppiness adapts to regime automatically (no manual switching)
- 4h HMA provides bias without over-filtering (single HTF = fewer conflicts)
- Session filter reduces noise trades (8-20 UTC = 12h window = ~50% of bars)
- Targets 40-80 trades/year on 1h = optimal fee/trade balance

Key difference from failed #1630, #1635, #1638:
- Fisher Transform instead of RSI (better reversal detection)
- Single HTF (4h) instead of dual (4h+12h) = less conflict, MORE TRADES
- CHOP threshold 55/45 (not 61.8/38.2) = more regime transitions = more trades
- ATR ratio > 1.5 (not > 2.0) = catches more vol spikes
- Session 8-20 UTC (not strict) = still allows decent trade frequency

Timeframe: 1h (required for this experiment)
HTF: 4h HMA via mtf_data.get_htf_data() — called ONCE before loop
Target: Sharpe > 0.618, trades > 30/symbol train, > 5/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_4h_hma_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - detects reversals in trending markets
    Transforms price into Gaussian distribution for clearer signals
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    for i in range(period, n):
        # Calculate HL2 (typical price)
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        # Avoid division by zero
        if highest - lowest < 1e-10:
            continue
        
        # Normalize price to 0-1 range
        normalized = (hl2 - lowest) / (highest - lowest)
        
        # Clamp to avoid extreme values
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > period:
            fisher_prev[i] = fisher[i - 1]
    
    return fisher, fisher_prev

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - identifies ranging vs trending markets
    CHOP > 61.8 = choppy/ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest - lowest < 1e-10:
            chop[i] = 100.0
            continue
        
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j == i - period + 1:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j - 1]), 
                        abs(low[j] - close[j - 1]))
            atr_sum += tr
        
        # CHOP formula
        if atr_sum < 1e-10:
            chop[i] = 100.0
        else:
            chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

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

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness_index(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Calculate ATR ratio for vol spike detection
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_ratio = np.full(n, np.nan)
    mask = ~np.isnan(atr_7) & ~np.isnan(atr_30) & (atr_30 > 1e-10)
    atr_ratio[mask] = atr_7[mask] / atr_30[mask]
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h timeframe (more trades)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]):
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
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = ranging (mean reversion mode)
        # CHOP < 45 = trending (trend follow mode)
        # 45-55 = neutral (allow both)
        is_range = chop[i] > 55.0
        is_trend = chop[i] < 45.0
        
        # === TREND BIAS (4h HMA) ===
        daily_bull = close[i] > hma_4h_aligned[i]
        daily_bear = close[i] < hma_4h_aligned[i]
        
        # === VOL SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 1.5 if not np.isnan(atr_ratio[i]) else False
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_short = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # LONG conditions (need 2+ confluence):
        # 1. Fisher long cross OR (range regime + price < HMA)
        # 2. 4h HMA bull OR neutral regime
        # 3. In session (8-20 UTC)
        long_score = 0
        if fisher_long:
            long_score += 2  # Strong signal
        if is_range and close[i] < hma_4h_aligned[i]:
            long_score += 1  # Mean reversion in range
        if daily_bull:
            long_score += 1  # Trend alignment
        if vol_spike:
            long_score += 1  # Vol spike reversal
        
        # SHORT conditions (need 2+ confluence):
        short_score = 0
        if fisher_short:
            short_score += 2  # Strong signal
        if is_range and close[i] > hma_4h_aligned[i]:
            short_score += 1  # Mean reversion in range
        if daily_bear:
            short_score += 1  # Trend alignment
        if vol_spike:
            short_score += 1  # Vol spike reversal
        
        # Entry threshold: need score >= 2 AND in session
        if in_session and long_score >= 2:
            desired_signal = BASE_SIZE
        elif in_session and short_score >= 2:
            desired_signal = -BASE_SIZE
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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