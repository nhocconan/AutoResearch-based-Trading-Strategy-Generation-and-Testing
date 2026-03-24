#!/usr/bin/env python3
"""
Experiment #1670: 1h Primary + 4h/12h HTF — Fisher Transform Mean Reversion with Regime Filter

Hypothesis: Previous 1h strategies (#1660, #1665, #1668) failed with Sharpe=0.000 due to 
OVER-FILTERING (too many confluence requirements = 0 trades). This strategy uses:

1. Ehlers Fisher Transform (period=9) for entry timing — better reversal signals than CRSI
   Long: Fisher crosses above -1.5 (oversold reversal)
   Short: Fisher crosses below +1.5 (overbought reversal)

2. 4h HMA(21) for immediate trend direction — faster than 12h for 1h entries
3. 12h HMA(21) for regime bias — bull/bear market filter
4. Volume filter: only 0.5x average (not 0.8x) to ensure trade generation
5. Session filter: 8-20 UTC ONLY (reduces trades by ~60%)

Key differences from failed 1h attempts:
- Fisher Transform instead of CRSI (more signals, less extreme thresholds)
- Single HTF trend (4h) + single regime (12h) — not 3+ HTF layers
- Volume threshold relaxed to 0.5x (not 0.8x or 1.0x)
- Asymmetric sizing: 0.25 with 12h trend, 0.15 against

Entry Logic:
- Fisher < -1.5 AND crossing up + 4h HMA bull + 12h bull = LONG 0.25
- Fisher > +1.5 AND crossing down + 4h HMA bear + 12h bear = SHORT 0.25
- Against 12h trend: reduce size to 0.15
- Session: only 8-20 UTC (reduces overnight noise)
- Volume: current > 0.5x 20-period average

Risk: 2.0x ATR trailing stop, discrete signal levels (0.0, ±0.15, ±0.25)
Target: 40-80 trades/year, Sharpe > 0.618, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_4h12h_session_volume_v2"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian normal distribution for clearer reversal signals
    
    Formula:
    1. Normalize price: (2 * (close - lowest_low) / (highest_high - lowest_low)) - 1
    2. Smooth with EMA
    3. Fisher = 0.5 * ln((1 + value) / (1 - value))
    
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    Exit: Fisher crosses 0.0 (neutral)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)  # previous bar fisher for crossover detection
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            fisher[i] = 0.0
            if i > period - 1:
                fisher_signal[i] = fisher[i - 1]
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 2.0 * (close[i] - lowest_low) / (highest_high - lowest_low) - 1.0
        
        # Clamp to avoid division by zero in log
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > period - 1:
            fisher_signal[i] = fisher[i - 1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # Combine
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
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

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_avg[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_avg

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = np.zeros(len(open_time_array), dtype=int)
    for i in range(len(open_time_array)):
        # Convert ms to seconds, then to datetime
        ts_seconds = open_time_array[i] / 1000.0
        # Extract hour (UTC)
        hours[i] = int((ts_seconds % 86400) / 3600)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for regime bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER (current > 0.5x average) ===
        volume_ok = volume[i] > 0.5 * vol_avg[i]
        
        # === HTF TREND BIAS ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher was below -1.5, now crossing above
        fisher_long = (fisher_signal[i] < -1.5) and (fisher[i] > fisher_signal[i])
        
        # Short: Fisher was above +1.5, now crossing below
        fisher_short = (fisher_signal[i] > 1.5) and (fisher[i] < fisher_signal[i])
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Only trade during session and with volume
        if in_session and volume_ok:
            # LONG entry
            if fisher_long:
                if hma_12h_bull:
                    # With 12h trend = full size
                    if hma_4h_bull:
                        desired_signal = BASE_SIZE
                    else:
                        desired_signal = REDUCED_SIZE
                elif hma_12h_bear:
                    # Against 12h trend = reduced size, only if 4h confirms
                    if hma_4h_bull:
                        desired_signal = REDUCED_SIZE
            
            # SHORT entry
            elif fisher_short:
                if hma_12h_bear:
                    # With 12h trend = full size
                    if hma_4h_bear:
                        desired_signal = -BASE_SIZE
                    else:
                        desired_signal = -REDUCED_SIZE
                elif hma_12h_bull:
                    # Against 12h trend = reduced size, only if 4h confirms
                    if hma_4h_bear:
                        desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
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