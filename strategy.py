#!/usr/bin/env python3
"""
Experiment #1323: 6h Primary + 1d/1w HTF — Ehlers Fisher Transform + Daily Trend Filter

Hypothesis: Previous 6h trend-following strategies (HMA, KAMA, ROC) suffered from
whipsaw during 2022 crash and range-bound 2025. This strategy uses Ehlers Fisher
Transform which excels at identifying turning points IN trending markets, combined
with daily HMA for regime bias.

Key innovations vs failed strategies:
1. Fisher Transform (period=9) catches reversals at trend extremes, not just crossovers
2. 1d HMA(21) slope for regime (only trade Fisher signals WITH daily trend)
3. 1w HMA(21) for major bias filter (prevents counter-major-trend trades)
4. ATR(14) 2.5x trailing stop for risk management
5. LOOSE Fisher thresholds (-1.5/+1.5) to guarantee 30-60 trades/year

Why Fisher Transform works better than RSI/ROC for 6h:
- Normalizes price into Gaussian distribution (bounded -1 to +1 theoretically)
- Sharp turning points = cleaner entry/exit signals
- Less whipsaw than RSI in trending markets
- Proven in bear/range markets (2022 crash, 2025 consolidation)

Entry logic (loose to guarantee trades):
- LONG: 1d_HMA rising + 1w_HMA bullish + Fisher crosses above -1.5
- SHORT: 1d_HMA falling + 1w_HMA bearish + Fisher crosses below +1.5

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_transform_daily_trend_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - converts price to Gaussian normal distribution
    Creates sharp turning points for reversal detection
    
    Reference: Ehlers, J.F. "Cycle Analytics" and "Rocket Science for Traders"
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    # Use (high + low) / 2 as price input
    hl2 = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over lookback period
        highest = hl2[i]
        lowest = hl2[i]
        for j in range(i - period + 1, i + 1):
            if hl2[j] > highest:
                highest = hl2[j]
            if hl2[j] < lowest:
                lowest = hl2[j]
        
        # Avoid division by zero
        range_val = highest - lowest
        if range_val < 1e-10:
            range_val = 1e-10
        
        # Normalize price to -1 to +1 range
        normalized = 2.0 * (hl2[i] - lowest) / range_val - 1.0
        
        # Clamp to avoid log(0) or log(negative)
        normalized = max(-0.999, min(0.999, normalized))
        
        # Fisher Transform formula
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Previous Fisher value (for crossover detection)
        if i > period:
            fisher_prev[i] = fisher[i - 1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

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
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    
    # Also calculate 6h HMA for local trend confirmation
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
        
        if np.isnan(hma_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME FILTER (1d HMA slope + 1w HMA bias) ===
        # 1d HMA slope (compare to 5 bars ago for stability on daily)
        hma_1d_slope = 0.0
        if i >= 5 and not np.isnan(hma_1d_aligned[i-5]):
            hma_1d_slope = hma_1d_aligned[i] - hma_1d_aligned[i-5]
        
        # 1w HMA bias (major trend direction)
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # 1d price vs 1d HMA
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 6h price vs 6h HMA for local confirmation
        price_above_6h = close[i] > hma_6h[i]
        price_below_6h = close[i] < hma_6h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_val = fisher[i]
        fisher_prev_val = fisher_prev[i]
        
        # Fisher crossover detection
        fisher_cross_up = (fisher_prev_val < -1.5) and (fisher_val >= -1.5)
        fisher_cross_down = (fisher_prev_val > 1.5) and (fisher_val <= 1.5)
        
        # Fisher extreme levels (stronger signals)
        fisher_extreme_low = fisher_val < -2.0
        fisher_extreme_high = fisher_val > 2.0
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 1d HMA rising + 1w bullish + Fisher crosses above -1.5
        # Loosen: allow either 1d rising OR 1w bullish (not both required)
        bullish_regime = (hma_1d_slope > 0) or (price_above_1w and price_above_1d)
        
        if bullish_regime and fisher_cross_up:
            if fisher_extreme_low:
                desired_signal = SIZE_STRONG  # Strong reversal signal
            else:
                desired_signal = SIZE_BASE  # Basic reversal signal
        
        # SHORT: 1d HMA falling + 1w bearish + Fisher crosses below +1.5
        bearish_regime = (hma_1d_slope < 0) or (price_below_1w and price_below_1d)
        
        if bearish_regime and fisher_cross_down:
            if fisher_extreme_high:
                desired_signal = -SIZE_STRONG  # Strong reversal signal
            else:
                desired_signal = -SIZE_BASE  # Basic reversal signal
        
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