#!/usr/bin/env python3
"""
Experiment #005: 1h Primary + 4h/1d HTF — Fisher Transform Reversals with Regime Filter

Hypothesis: Fisher Transform catches reversals better than CRSI in bear/range markets (2022 crash, 2025 bear).
Combined with 4h HMA trend direction + 1d bias + volume/session filters = fewer but higher quality trades.

Key innovations:
1. Ehlers Fisher Transform (period=9) - normalizes price to Gaussian, better reversal detection
2. 4h HMA for immediate trend direction (not hard filter, but sizing weight)
3. 1d HMA for regime bias (bull/bear market context)
4. Volume filter: only trade when volume > 0.8x 20-period average
5. Session filter: only 8-20 UTC (high liquidity, less manipulation)
6. Asymmetric sizing: 0.30 with HTF trend, 0.20 against HTF trend
7. ATR 2.5x trailing stoploss

Why 1h with HTF: 4h/1d determine DIRECTION, 1h determines ENTRY TIMING.
This gives HTF trade frequency (30-60/year) with lower TF execution precision.

Target: Sharpe > 0.15, trades > 30/symbol train, > 3/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h1d_hma_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian-normalized value for clearer reversal signals
    
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    Research shows superior reversal detection in bear/range markets vs RSI
    """
    n = len(close)
    if n < period + 10:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)  # 1-period lagged fisher for crossover detection
    
    # Calculate typical price and normalize
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            continue
        
        # Normalize price to 0-1 range
        price_normalized = (close[i] - lowest_low) / (highest_high - lowest_low)
        
        # Clamp to avoid division issues
        price_normalized = np.clip(price_normalized, 0.001, 0.999)
        
        # Calculate value (Ehlers formula)
        value = 0.66 * ((price_normalized - 0.5) / 0.5) + 0.67 * np.sinh(
            0.66 * ((price_normalized - 0.5) / 0.5)
        ) if i > period else 0.0
        
        # Smooth value
        if i > period:
            value = 0.5 * value + 0.5 * (0.66 * ((price_normalized - 0.5) / 0.5) + 
                       0.67 * np.sinh(0.66 * ((price_normalized - 0.5) / 0.5)))
        
        # Convert to Fisher value
        if abs(value) >= 0.999:
            value = np.sign(value) * 0.999
        
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        
        if i > period:
            fisher_signal[i] = fisher[i - 1]
    
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
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
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

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_hour_from_timestamp(open_time):
    """Extract UTC hour from timestamp (milliseconds)"""
    # Convert milliseconds to seconds, then to datetime
    timestamps_sec = open_time / 1000.0
    hours = ((timestamps_sec % 86400) / 3600).astype(int)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values if "open_time" in prices.columns else np.arange(len(close))
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1 - CRITICAL) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for regime bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === CALCULATE PRIMARY (1h) INDICATORS ===
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    hours = get_hour_from_timestamp(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing levels (discrete to minimize fee churn)
    SIZE_WITH_TREND = 0.30
    SIZE_AGAINST_TREND = 0.20
    MAX_SIZE = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === HTF TREND DIRECTION ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF REGIME BIAS ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        fisher_long = False
        fisher_short = False
        
        # Long: Fisher crosses above -1.5 (oversold reversal)
        if fisher_signal[i] < -1.5 and fisher[i] >= -1.5:
            fisher_long = True
        
        # Short: Fisher crosses below +1.5 (overbought reversal)
        if fisher_signal[i] > 1.5 and fisher[i] <= 1.5:
            fisher_short = True
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if in_session and volume_ok:
            if fisher_long:
                # Long signal - check HTF alignment
                if hma_4h_bull and hma_1d_bull:
                    # All aligned bullish - full size
                    desired_signal = SIZE_WITH_TREND
                elif hma_4h_bull or hma_1d_bull:
                    # Partial alignment - reduced size
                    desired_signal = SIZE_AGAINST_TREND
                else:
                    # Against both HTF - skip or very small
                    desired_signal = 0.0
            
            elif fisher_short:
                # Short signal - check HTF alignment
                if hma_4h_bear and hma_1d_bear:
                    # All aligned bearish - full size
                    desired_signal = -SIZE_WITH_TREND
                elif hma_4h_bear or hma_1d_bear:
                    # Partial alignment - reduced size
                    desired_signal = -SIZE_AGAINST_TREND
                else:
                    # Against both HTF - skip or very small
                    desired_signal = 0.0
        
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
        elif desired_signal >= SIZE_AGAINST_TREND * 0.85:
            final_signal = SIZE_AGAINST_TREND
        elif desired_signal <= -SIZE_AGAINST_TREND * 0.85:
            final_signal = -SIZE_AGAINST_TREND
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