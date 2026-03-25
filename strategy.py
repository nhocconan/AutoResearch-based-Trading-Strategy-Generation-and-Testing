#!/usr/bin/env python3
"""
Experiment #1267: 6h Primary + 1d/1w HTF — Fisher Transform Reversal + HTF Trend

Hypothesis: The Ehlers Fisher Transform excels at catching reversals in bear/range markets
(like 2025 BTC/ETH). Combined with 1d/1w trend bias, this should generate 30-60 trades/year
with high win rate. Key differences from failed 6h strategies:

1. Fisher Transform (period=9) - normalizes price to Gaussian, extremes at ±2.0 signal reversals
2. 1d HMA(21) + 1w HMA(21) for major trend bias (only trade with HTF direction)
3. NO ADX filter - too restrictive, caused 0 trades in prior experiments
4. LOOSE Fisher thresholds (-1.5/+1.5 for entry, -2.0/+2.0 for strong) to guarantee trades
5. ATR(14) 2.5x trailing stop for risk management

Why this should work on 6h:
- Fisher Transform proven in research for bear market reversals (75% win rate)
- 6h = natural 30-60 trades/year (between noisy 4h and slow 12h)
- Dual HTF (1d+1w) = strong directional bias without over-filtering
- No conflicting regime filters = conditions can actually align
- Discrete sizing (0.0, ±0.25, ±0.30) = minimal fee churn

Entry logic (LOOSE to guarantee 30+ trades):
- LONG: 1d_HMA bullish + 1w_HMA bullish + Fisher < -1.5 + turning up
- SHORT: 1d_HMA bearish + 1w_HMA bearish + Fisher > +1.5 + turning down

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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Extremes at ±2.0 signal potential reversals
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate median price (HL2)
    hl2 = (high + low) / 2.0
    
    # Track highest high and lowest low over period
    for i in range(period - 1, n):
        # Get price range over lookback period
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        if highest == lowest:
            continue
        
        # Normalize price to range [-1, 1]
        price_norm = 0.6667 * ((hl2[i] - lowest) / (highest - lowest) - 0.5)
        price_norm = max(-0.999, min(0.999, price_norm))  # Clamp to avoid log errors
        
        # Fisher calculation
        fisher_val = 0.5 * np.log((1.0 + price_norm) / (1.0 - price_norm))
        
        # Smooth with previous value (Ehlers method)
        if i > period - 1 and not np.isnan(fisher[i-1]):
            fisher_val = 0.5 * fisher_val + 0.5 * fisher[i-1]
        
        fisher[i] = fisher_val
        if i > 0 and not np.isnan(fisher[i-1]):
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

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
    fisher_6h, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    
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
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher_6h[i]) or np.isnan(fisher_trigger[i]):
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
        
        # === TREND DIRECTION (HTF HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # Strong trend: both 1d and 1w agree
        trend_bullish = price_above_1d and price_above_1w
        trend_bearish = price_below_1d and price_below_1w
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_curr = fisher_6h[i]
        fisher_prev = fisher_trigger[i]  # Previous bar's fisher value
        
        # Fisher turning up (bullish reversal signal)
        fisher_turning_up = fisher_prev < fisher_curr
        # Fisher turning down (bearish reversal signal)
        fisher_turning_down = fisher_prev > fisher_curr
        
        # === ENTRY LOGIC (LOOSE - guarantee 30+ trades/year) ===
        desired_signal = 0.0
        
        # LONG: HTF bullish + Fisher oversold + turning up
        # Threshold: -1.5 for basic entry, -2.0 for strong entry
        if trend_bullish:
            if fisher_curr < -1.5 and fisher_turning_up:
                if fisher_curr < -2.0:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: HTF bearish + Fisher overbought + turning down
        # Threshold: +1.5 for basic entry, +2.0 for strong entry
        elif trend_bearish:
            if fisher_curr > 1.5 and fisher_turning_down:
                if fisher_curr > 2.0:
                    desired_signal = -SIZE_STRONG
                else:
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