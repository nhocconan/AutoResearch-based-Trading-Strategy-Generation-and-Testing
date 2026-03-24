#!/usr/bin/env python3
"""
Experiment #1620: 1h Primary + 4h/12h HTF — Fisher Transform + HMA Regime + Session Filter

Hypothesis: After 11 failed 4h experiments and multiple 1h failures, the key is using
HTF (4h/12h) for SIGNAL DIRECTION and 1h only for ENTRY TIMING. This gives HTF
trade frequency with 1h execution precision.

Key innovations:
1. EHLERS FISHER TRANSFORM (period=9) - proven reversal catcher in bear markets
   Long when Fisher crosses above -1.5, Short when crosses below +1.5
2. 4h HMA for immediate trend bias (faster than 1d, better for 1h entries)
3. 12h HMA for regime filter (avoid counter-trend in strong moves)
4. SESSION FILTER (8-20 UTC) - only trade during high-volume hours
5. VOLUME FILTER (>0.8x 20-bar avg) - confirm participation
6. DISCRETE sizing: 0.0, ±0.20, ±0.25 — minimize fee churn
7. STRICT trade frequency: target 30-60 trades/year for 1h

Why this should beat Sharpe 0.618:
- Fisher Transform catches reversals in bear/range markets (2025 test period)
- 4h/12h HMA confluence prevents counter-trend trades
- Session filter avoids low-volume whipsaws (Asian session)
- Volume confirmation ensures real moves, not noise
- 1h targets 30-60 trades/year — optimal fee/trade balance for lower TF
- ATR stoploss (2.5x) protects against crashes

Timeframe: 1h (required for this experiment)
HTF: 4h + 12h HMA for bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
Trade Frequency: 30-60/year (CRITICAL — too many = fee drag kills profit)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_4h12h_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals in bear/range markets
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    for i in range(period, n):
        # Calculate (2 * price - HH - LL) / (HH - LL)
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            x = (2.0 * close[i] - highest_high - lowest_low) / price_range
            # Clamp to [-0.999, 0.999] to avoid log(0)
            x = np.clip(x, -0.999, 0.999)
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
            
            # Previous value for crossover detection
            if i > period:
                prev_highest = np.max(high[i-period:i])
                prev_lowest = np.min(low[i-period:i])
                prev_range = prev_highest - prev_lowest
                if prev_range > 1e-10:
                    x_prev = (2.0 * close[i-1] - prev_highest - prev_lowest) / prev_range
                    x_prev = np.clip(x_prev, -0.999, 0.999)
                    fisher_prev[i] = 0.5 * np.log((1.0 + x_prev) / (1.0 - x_prev))
    
    return fisher, fisher_prev

def calculate_hma(close, period=21):
    """Hull Moving Average - responsive trend indicator"""
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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
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
    
    # Calculate and align 4h HMA for immediate trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for regime filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    
    # 1h HMA for local trend
    hma_1h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.22  # Smaller for 1h to reduce fee impact
    
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
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma[i]) or np.isnan(hma_1h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (>0.8x 20-bar avg) ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === TREND BIAS (4h HMA) ===
        hma4h_bull = close[i] > hma_4h_aligned[i]
        hma4h_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (12h HMA) ===
        hma12h_bull = close[i] > hma_12h_aligned[i]
        hma12h_bear = close[i] < hma_12h_aligned[i]
        
        # === TREND CONFLUENCE (4h and 12h agree) ===
        strong_bull = hma4h_bull and hma12h_bull
        strong_bear = hma4h_bear and hma12h_bear
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # LONG: Fisher reversal + strong bull trend + session + volume
        if fisher_long and strong_bull and in_session and volume_ok:
            desired_signal = BASE_SIZE
        
        # SHORT: Fisher reversal + strong bear trend + session + volume
        elif fisher_short and strong_bear and in_session and volume_ok:
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