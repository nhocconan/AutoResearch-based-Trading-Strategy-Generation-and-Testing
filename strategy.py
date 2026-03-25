#!/usr/bin/env python3
"""
Experiment #1500: 6h Primary + 1d/1w HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: 6h timeframe is underexplored (0 experiments). Fisher Transform excels at
catching reversals in bear/range markets (2022 crash, 2025 bear). KAMA adapts to
volatility regimes automatically. Combined with 1d/1w HMA bias, this should work
across all market conditions.

Key components:
1. 1w HMA(21) for major trend bias (very slow, only changes quarterly)
2. 1d HMA(21) for intermediate trend direction
3. 6h Fisher Transform(9) for entry timing (crosses -1.5/+1.5 levels)
4. 6h KAMA(10) for adaptive trend confirmation
5. 6h ATR(14) for stoploss (2.5x ATR trailing)
6. LOOSE entry thresholds to guarantee ≥30 trades/year

Why Fisher Transform:
- Normalizes price into bounded range (-2 to +2 typically)
- Sharp reversals at extremes catch bear market bounces
- Proven in Ehlers literature for cycle detection
- Works well when trend strategies fail (2022, 2025)

Why KAMA:
- Efficiency Ratio adapts smoothing based on trend strength
- Less whipsaw than EMA in choppy markets
- Automatically adjusts to volatility regimes

Entry logic (LOOSE - must generate trades):
- LONG: 1w_HMA bullish OR 1d_HMA bullish + Fisher cross above -1.5 + KAMA rising
- SHORT: 1w_HMA bearish OR 1d_HMA bearish + Fisher cross below +1.5 + KAMA falling
- Use OR not AND for HTF filters (too many ANDs = 0 trades)

Target: Sharpe>0.6, trades>=120 train (30/year), trades>=10 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_kama_hma_1d1w_loose_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        if price_change < 1e-10:
            er[i] = 0.0
        else:
            volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            if volatility > 1e-10:
                er[i] = price_change / volatility
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price into bounded range for reversal detection
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        # Find highest high and lowest low over period
        highest = np.max(close[i - period + 1:i + 1])
        lowest = np.min(close[i - period + 1:i + 1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            fisher[i] = 0.0
        else:
            # Normalize price to -1 to +1 range
            normalized = 0.6667 * ((close[i] - lowest) / price_range - 0.5) + 0.67 * fisher[i - 1] if i > period - 1 and not np.isnan(fisher[i - 1]) else 0.0
            normalized = np.clip(normalized, -0.999, 0.999)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
            fisher_signal[i] = fisher[i - 1] if i > 0 else 0.0
    
    return fisher, fisher_signal

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
    fisher, fisher_signal = calculate_fisher(close, period=9)
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # KAMA slope (rising/falling)
    kama_slope = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if not np.isnan(kama_10[i]) and not np.isnan(kama_10[i-1]):
            kama_slope[i] = kama_10[i] - kama_10[i-1]
    
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
        
        if np.isnan(fisher[i]) or np.isnan(kama_10[i]) or np.isnan(kama_slope[i]):
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
        
        # === TREND BIAS (1w and 1d HMA) ===
        # Use OR logic - too many AND filters = 0 trades
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_val = fisher[i]
        fisher_prev = fisher_signal[i] if not np.isnan(fisher_signal[i]) else fisher_val
        
        # Fisher crosses above -1.5 (bullish reversal from oversold)
        fisher_cross_up = fisher_prev < -1.5 and fisher_val >= -1.5
        # Fisher crosses below +1.5 (bearish reversal from overbought)
        fisher_cross_down = fisher_prev > 1.5 and fisher_val <= 1.5
        # Fisher extreme oversold (strong long signal)
        fisher_extreme_low = fisher_val < -1.8
        # Fisher extreme overbought (strong short signal)
        fisher_extreme_high = fisher_val > 1.8
        
        # === KAMA SLOPE (trend confirmation) ===
        kama_rising = kama_slope[i] > 0
        kama_falling = kama_slope[i] < 0
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG entries (multiple paths to ensure trades)
        long_score = 0
        
        # Path 1: Fisher extreme oversold (strongest signal)
        if fisher_extreme_low:
            long_score += 2
        
        # Path 2: Fisher cross up from oversold
        if fisher_cross_up:
            long_score += 2
        
        # Path 3: Price above 1w HMA (major trend bullish)
        if price_above_1w:
            long_score += 1
        
        # Path 4: Price above 1d HMA (intermediate trend bullish)
        if price_above_1d:
            long_score += 1
        
        # Path 5: KAMA rising (momentum confirmation)
        if kama_rising:
            long_score += 1
        
        # LONG if score >= 3 (flexible confluence)
        if long_score >= 3:
            desired_signal = SIZE_STRONG if long_score >= 4 else SIZE_BASE
        
        # SHORT entries (multiple paths to ensure trades)
        short_score = 0
        
        # Path 1: Fisher extreme overbought (strongest signal)
        if fisher_extreme_high:
            short_score += 2
        
        # Path 2: Fisher cross down from overbought
        if fisher_cross_down:
            short_score += 2
        
        # Path 3: Price below 1w HMA (major trend bearish)
        if price_below_1w:
            short_score += 1
        
        # Path 4: Price below 1d HMA (intermediate trend bearish)
        if price_below_1d:
            short_score += 1
        
        # Path 5: KAMA falling (momentum confirmation)
        if kama_falling:
            short_score += 1
        
        # SHORT if score >= 3 (flexible confluence)
        if short_score >= 3:
            desired_signal = -SIZE_STRONG if short_score >= 4 else -SIZE_BASE
        
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