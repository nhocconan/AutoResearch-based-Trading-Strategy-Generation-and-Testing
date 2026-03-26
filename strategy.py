#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian Breakout + Williams %R + Volume Confirmation + 1w Trend

HYPOTHESIS: 12h Donchian(20) channel breakouts mark institutional moves. 
Williams %R confirms momentum at the breakout point (not oversold/overbought extremes).
Volume spike confirms institutional participation. 1w HMA trend filters counter-trend entries.

WHY THIS WORKS IN BOTH BULL AND BEAR:
- Bull: Long breakouts of 20-period high + Williams %R > -20 (momentum confirming)
- Bear: Short breakouts of 20-period low + Williams %R < -80 (momentum confirming)
- 1w HMA trend keeps us aligned with macro direction
- 12h timeframe naturally limits trades to ~50-150 over 4 years

TARGET: 50-150 total over 4 years (~12-37/year). HARD MAX: 200.
DB reference: Donchian + HMA + volume + ATR (SOL: test Sharpe 1.10-1.38)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_williams_vol_1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    williams = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and not np.isnan(close[i]):
            williams[i] = -100.0 * (highest - close[i]) / (highest - lowest)
    
    return williams

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
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for trend direction
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Align 1w highs/lows for trend check
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, df_1w['close'].values)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    williams_r = calculate_williams_r(high, low, close, period=14)
    
    # Donchian channels (20 periods on 12h = 10 days)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume MA and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = donchian_period + 30
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(williams_r[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND CHECK (1w HMA) ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        trend_bullish = price_above_1w_hma
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Upper breakout: price breaks above 20-period high
        upper_breakout = close[i] > donchian_high[i]
        # Lower breakout: price breaks below 20-period low
        lower_breakout = close[i] < donchian_low[i]
        
        # Previous bar was NOT in breakout (avoid repeated signals)
        prev_upper_breakout = close[i-1] > donchian_high[i-1] if i > warmup else False
        prev_lower_breakout = close[i-1] < donchian_low[i-1] if i > warmup else False
        
        new_upper_breakout = upper_breakout and not prev_upper_breakout
        new_lower_breakout = lower_breakout and not prev_lower_breakout
        
        # === WILLIAMS %R CONFIRMATION ===
        # Long: Williams %R above -50 (momentum not exhausted) but not extreme
        # Short: Williams %R below -50 (momentum not exhausted) but not extreme
        williams_ok_long = williams_r[i] > -50 and williams_r[i] < -10
        williams_ok_short = williams_r[i] < -50 and williams_r[i] > -90
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: New upper breakout + bullish 1w trend + Williams %R confirms + volume spike
            if new_upper_breakout and trend_bullish and williams_ok_long and vol_spike:
                desired_signal = SIZE
            
            # SHORT: New lower breakout + bearish 1w trend + Williams %R confirms + volume spike
            if new_lower_breakout and (not trend_bullish) and williams_ok_short and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === EXIT ON OPPOSITE DONCHIAN ===
        # Exit long if price breaks lower channel (trend reversal)
        exit_long = position_side > 0 and close[i] < donchian_low[i]
        # Exit short if price breaks upper channel (trend reversal)
        exit_short = position_side < 0 and close[i] > donchian_high[i]
        
        if exit_long or exit_short:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals