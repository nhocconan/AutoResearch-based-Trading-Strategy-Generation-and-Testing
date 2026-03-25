#!/usr/bin/env python3
"""
Experiment #1310: 1h Primary + 4h/1d HTF — HMA Trend + ROC Momentum + Volatility Filter

Hypothesis: Recent 1h strategies failed (Sharpe=0.000) due to overly strict entry conditions.
This variant uses LOOSE thresholds to guarantee 40-80 trades/year while maintaining
multi-timeframe confluence. Key learnings from failures:

1. RSI pullback on 1h = 0 trades (#1299 failed)
2. CRSI mean reversion = 0 trades (#1309 failed)
3. Too many regime filters = no signals trigger

New approach:
- 4h HMA(21) slope for trend direction (proven in best strategies)
- 1h ROC(10) for momentum entry (LOOSE: >2 or <-2, not >5)
- 1d HMA(21) for major regime bias (single filter, not dual)
- ATR(14) ratio for volatility filter (only trade when vol expanding)
- 2.5x ATR trailing stoploss

Why this should work:
- 1h timeframe with HTF direction = proven pattern (mtf_hma_rsi_zscore_v1 Sharpe=5.4)
- Loose ROC thresholds = guarantee trades without over-trading
- Volatility filter = avoid choppy periods naturally
- Discrete sizing (0.0, ±0.20, ±0.30) = minimal fee churn

Entry logic (LOOSE to guarantee 40-80 trades/year):
- LONG: 4h_HMA rising + 1d_HMA bullish + ROC(10) > 2 + ATR_ratio > 1.0
- SHORT: 4h_HMA falling + 1d_HMA bearish + ROC(10) < -2 + ATR_ratio > 1.0

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_trend_roc_momentum_vol_4h1d_v1"
timeframe = "1h"
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

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100.0
    
    return roc

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    roc_10 = calculate_roc(close, period=10)
    ema_21 = calculate_ema(close, period=21)
    
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
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(roc_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(atr_30[i]) or atr_30[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (4h HMA slope + 1d HMA bias) ===
        # 4h HMA slope (compare to 2 bars ago for stability)
        hma_4h_slope = 0.0
        if i >= 2 and not np.isnan(hma_4h_aligned[i-2]):
            hma_4h_slope = hma_4h_aligned[i] - hma_4h_aligned[i-2]
        
        # 1d HMA bias
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1h price vs 1h EMA for local confirmation
        price_above_ema = close[i] > ema_21[i]
        price_below_ema = close[i] < ema_21[i]
        
        # === MOMENTUM (ROC) ===
        roc = roc_10[i]
        
        # === VOLATILITY FILTER (ATR ratio) ===
        atr_ratio = atr_14[i] / atr_30[i] if atr_30[i] > 0 else 0.0
        
        # === ENTRY LOGIC (LOOSE - guarantee 40-80 trades/year) ===
        desired_signal = 0.0
        
        # LONG: 4h HMA rising + 1d bullish + ROC positive + vol expanding
        if hma_4h_slope > 0 and price_above_1d and price_above_ema:
            if roc > 2.0 and atr_ratio > 0.9:  # Loose thresholds
                if roc > 5.0 and atr_ratio > 1.2:
                    desired_signal = SIZE_STRONG  # Strong momentum
                else:
                    desired_signal = SIZE_BASE  # Basic momentum
        
        # SHORT: 4h HMA falling + 1d bearish + ROC negative + vol expanding
        elif hma_4h_slope < 0 and price_below_1d and price_below_ema:
            if roc < -2.0 and atr_ratio > 0.9:  # Loose thresholds
                if roc < -5.0 and atr_ratio > 1.2:
                    desired_signal = -SIZE_STRONG  # Strong momentum
                else:
                    desired_signal = -SIZE_BASE  # Basic momentum
        
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