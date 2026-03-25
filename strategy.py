#!/usr/bin/env python3
"""
Experiment #1391: 6h Primary + 1d/1w HTF — Ehlers Fisher Transform + Super Smoother Trend

Hypothesis: Previous 6h strategies failed due to either (1) too many filters = 0 trades, 
or (2) wrong indicator type (RSI/CRSI mean reversion doesn't work on 6h).

This strategy uses PROVEN reversal indicators from Ehlers' research:
1. Fisher Transform (period=9) - Catches reversals in bear/range markets better than RSI
   Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
2. Ehlers Super Smoother (period=10) - Less lag than EMA, removes cycle noise
   Used for trend confirmation without whipsaw
3. 1d HMA(21) - Major trend bias (keep what works from prior experiments)
4. 1w HMA(21) - Weekly regime filter (only trade with weekly trend direction)

Why this should work where others failed:
- Fisher Transform specifically designed for reversal detection in non-trending markets
- Super Smoother has less lag than KAMA/EMA while filtering noise
- Fewer filters = more trades (previous 6h experiments had Sharpe=0.000 = no trades)
- 6h TF naturally produces 30-60 trades/year (fee-friendly)

Entry logic (LOOSE to guarantee trades):
- LONG: Fisher > -1.5 (crossing up) + SuperSmoother sloping up + price > 1d_HMA
- SHORT: Fisher < +1.5 (crossing down) + SuperSmoother sloping down + price < 1d_HMA
- Weekly HMA adds conviction but NOT required (prevents 0-trade problem)

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_supersmoother_trend_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - converts price to Gaussian distribution for reversal detection
    Reference: "Cycle Analytics for Traders" by John Ehlers
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate median price
    median_price = (high + low) / 2.0
    
    # Calculate normalized price (0.001 to 0.999 range)
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        if highest > lowest:
            normalized = (median_price[i] - lowest) / (highest - lowest)
            # Clamp to valid range for log calculation
            normalized = max(0.001, min(0.999, normalized))
            
            # Fisher calculation
            fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
            
            # Smooth with previous value (Ehlers formula)
            if i > period and not np.isnan(fisher[i-1]):
                fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
                fisher_prev[i] = fisher[i-1]
            else:
                fisher[i] = fisher_val
                fisher_prev[i] = fisher_val
    
    return fisher, fisher_prev

def calculate_super_smoother(close, period=10):
    """
    Ehlers Super Smoother Filter - removes cycle noise with minimal lag
    Reference: "Empirical Mode Decomposition" by John Ehlers
    """
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    ss = np.full(n, np.nan, dtype=np.float64)
    
    # Coefficients for Super Smoother
    a1 = np.exp(-np.sqrt(2) * np.pi / period)
    b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    # Initialize
    ss[0] = close[0]
    if n > 1:
        ss[1] = close[1]
    
    # Apply filter
    for i in range(2, n):
        if not np.isnan(close[i]) and not np.isnan(ss[i-1]) and not np.isnan(ss[i-2]):
            ss[i] = c1 * (close[i] + close[i-1]) / 2 + c2 * ss[i-1] + c3 * ss[i-2]
    
    return ss

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
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    supersmooth = calculate_super_smoother(close, period=10)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(supersmooth[i]):
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
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA for regime (adds conviction, not required)
        price_above_1w = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        price_below_1w = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === SUPER SMOOTHER TREND ===
        ss_uptrend = False
        ss_downtrend = False
        
        if i >= 3 and not np.isnan(supersmooth[i-1]) and not np.isnan(supersmooth[i-2]):
            # Super Smoother sloping up (2 consecutive higher values)
            if supersmooth[i] > supersmooth[i-1] > supersmooth[i-2]:
                ss_uptrend = True
            # Super Smoother sloping down
            elif supersmooth[i] < supersmooth[i-1] < supersmooth[i-2]:
                ss_downtrend = True
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_val = fisher[i]
        fisher_prev_val = fisher_prev[i]
        
        # Fisher crossing above -1.5 (long signal)
        fisher_long = fisher_prev_val < -1.5 and fisher_val >= -1.5
        
        # Fisher crossing below +1.5 (short signal)
        fisher_short = fisher_prev_val > 1.5 and fisher_val <= 1.5
        
        # Also allow Fisher in extreme zones for continuation
        fisher_oversold = fisher_val < -1.0
        fisher_overbought = fisher_val > 1.0
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: Fisher reversal/oversold + SuperSmoother up + price > 1d_HMA
        if (fisher_long or fisher_oversold) and ss_uptrend and price_above_1d:
            if price_above_1w:
                # Strong trend alignment (1d + 1w both bullish)
                desired_signal = SIZE_STRONG
            else:
                # Basic long (only 1d bullish)
                desired_signal = SIZE_BASE
        
        # SHORT: Fisher reversal/overbought + SuperSmoother down + price < 1d_HMA
        elif (fisher_short or fisher_overbought) and ss_downtrend and price_below_1d:
            if price_below_1w:
                # Strong trend alignment (1d + 1w both bearish)
                desired_signal = -SIZE_STRONG
            else:
                # Basic short (only 1d bearish)
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