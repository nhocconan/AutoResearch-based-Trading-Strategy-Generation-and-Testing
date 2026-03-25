#!/usr/bin/env python3
"""
Experiment #1331: 6h Primary + 1w/1d HTF — HMA Crossover + ROC Momentum

Hypothesis: The previous HMA slope approach was too restrictive, causing 0 trades on some symbols.
This variant uses HMA PRICE CROSSOVER (cleaner signals) instead of slope calculation, with
looser ROC thresholds to guarantee 30-60 trades/year. Key improvements:

1. 1w HMA(21) for MAJOR trend bias (only long if price > weekly HMA, only short if below)
2. 1d HMA(21) for intermediate confirmation (adds filter without over-constraining)
3. 6h price vs 6h HMA(21) crossover for entry timing (cleaner than slope)
4. ROC(10) with LOOSE threshold (>|2| instead of |>5|) = more trades
5. Volume confirmation: volume > 0.8 * SMA(volume, 20) = avoids low-liquidity traps
6. ATR(14) 2.5x trailing stop for risk management

Why this should beat Sharpe=0.447:
- Price crossover is cleaner than slope (fewer whipsaws in ranging markets)
- Weekly bias prevents counter-trend trades in major moves
- Loose ROC threshold guarantees trades on all symbols (no Sharpe=0.000)
- Volume filter avoids false breakouts on low activity bars
- 6h timeframe = natural 30-60 trades/year (fee-friendly)

Entry logic (LOOSE to guarantee trades on ALL symbols):
- LONG: price > 1w_HMA + price > 1d_HMA + price crosses above 6h_HMA + ROC > 2 + volume ok
- SHORT: price < 1w_HMA + price < 1d_HMA + price crosses below 6h_HMA + ROC < -2 + volume ok

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_crossover_roc_volume_1w1d_v1"
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

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        window = volume[i - period + 1:i + 1]
        if not np.any(np.isnan(window)):
            vol_sma[i] = np.mean(window)
    
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    roc_10 = calculate_roc(close, period=10)
    hma_6h = calculate_hma(close, period=21)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
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
    
    # Track previous bar for crossover detection
    prev_price_above_6h = None
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_price_above_6h = None
            continue
        
        if np.isnan(roc_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_price_above_6h = None
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_price_above_6h = None
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(vol_sma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_price_above_6h = None
            continue
        
        # === TREND DIRECTION (1w + 1d HMA bias) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 6h price vs 6h HMA for local confirmation
        price_above_6h = close[i] > hma_6h[i]
        price_below_6h = close[i] < hma_6h[i]
        
        # === CROSSOVER DETECTION ===
        crossover_long = False
        crossover_short = False
        
        if prev_price_above_6h is not None:
            # Bullish crossover: was below, now above
            if not prev_price_above_6h and price_above_6h:
                crossover_long = True
            # Bearish crossover: was above, now below
            if prev_price_above_6h and price_below_6h:
                crossover_short = True
        
        prev_price_above_6h = price_above_6h
        
        # === MOMENTUM (ROC) ===
        roc = roc_10[i]
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * vol_sma_20[i] if vol_sma_20[i] > 0 else False
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + 1d bullish + crossover + ROC positive + volume ok
        if price_above_1w and price_above_1d and crossover_long:
            if roc > 2.0:  # Very loose momentum threshold
                if volume_ok:
                    if roc > 6.0:
                        desired_signal = SIZE_STRONG  # Strong momentum
                    else:
                        desired_signal = SIZE_BASE  # Basic momentum
        
        # SHORT: 1w bearish + 1d bearish + crossover + ROC negative + volume ok
        elif price_below_1w and price_below_1d and crossover_short:
            if roc < -2.0:  # Very loose momentum threshold
                if volume_ok:
                    if roc < -6.0:
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