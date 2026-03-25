#!/usr/bin/env python3
"""
Experiment #1274: 1d Primary + 1w HTF — Simple HMA Trend + Weekly Bias

Hypothesis: After 1000+ failed experiments, the pattern is clear: COMPLEX = 0 TRADES.
Strategies with choppiness + CRSI + multiple HTF filters generate Sharpe=0.000 because
entry conditions never align. On 1d timeframe, we need SIMPLICITY:

1. 1w HMA(21) for MAJOR trend bias (only ONE HTF filter, not multiple)
2. 1d HMA(9) vs HMA(21) crossover for entries (classic, proven)
3. ROC(5) for loose momentum confirmation (threshold >2 or <-2, not >8)
4. ATR(14) 2.5x trailing stop for risk management
5. LOOSE conditions to guarantee 20-50 trades/year on daily

Why this should work:
- 1d timeframe = natural 20-50 trades/year (fee-friendly)
- Single HTF filter (1w) = directional bias without over-filtering
- HMA crossover = proven trend signal, generates regular trades
- Loose ROC threshold = catches moves early, doesn't wait for extreme
- Discrete sizing (0.0, ±0.25, ±0.30) = minimal fee churn

CRITICAL LESSON: The 12+ failed strategies all had 0 trades due to over-filtering.
This strategy uses MINIMAL filters to guarantee trade generation.

Entry logic (LOOSE):
- LONG: 1w_HMA bullish + 1d_HMA9 > 1d_HMA21 + ROC(5) > 2
- SHORT: 1w_HMA bearish + 1d_HMA9 < 1d_HMA21 + ROC(5) < -2

Target: Sharpe>0.5, trades>=20 train, trades>=5 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_crossover_weekly_bias_roc_v1"
timeframe = "1d"
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

def calculate_roc(close, period=5):
    """Rate of Change - momentum indicator (loose period for more signals)"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100.0
    
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    roc_5 = calculate_roc(close, period=5)
    
    # Daily HMA for crossover signals
    hma_1d_9 = calculate_hma(close, period=9)
    hma_1d_21 = calculate_hma(close, period=21)
    
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
    
    # Warmup period (shorter for 1d to get more trades)
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(roc_5[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_9[i]) or np.isnan(hma_1d_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY TREND BIAS (1w HMA slope) ===
        hma_1w_slope = 0.0
        if i >= 7 and not np.isnan(hma_1w_aligned[i-7]):
            hma_1w_slope = hma_1w_aligned[i] - hma_1w_aligned[i-7]
        
        weekly_bullish = hma_1w_slope > 0
        weekly_bearish = hma_1w_slope < 0
        
        # === DAILY HMA CROSSOVER ===
        hma_cross_bullish = hma_1d_9[i] > hma_1d_21[i]
        hma_cross_bearish = hma_1d_9[i] < hma_1d_21[i]
        
        # === MOMENTUM (ROC) - LOOSE THRESHOLD ===
        roc = roc_5[i]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: Weekly bullish + Daily HMA cross + ROC positive
        if weekly_bullish and hma_cross_bullish:
            if roc > 2.0:  # Very loose momentum threshold
                if roc > 5.0:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: Weekly bearish + Daily HMA cross + ROC negative
        elif weekly_bearish and hma_cross_bearish:
            if roc < -2.0:  # Very loose momentum threshold
                if roc < -5.0:
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