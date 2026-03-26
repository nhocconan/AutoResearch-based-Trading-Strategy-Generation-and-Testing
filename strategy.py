#!/usr/bin/env python3
"""
Experiment #024: 12h Donchian Momentum + Volume Confirmation + 1w Regime

HYPOTHESIS: 12h Donchian breakouts with consecutive closes outside the channel
capture institutional moves. Combined with volume confirmation and 1w HMA regime
filter, this should work in both bull markets (trend-following longs) and bear
markets (short rallies to the HMA). 12h timeframe reduces trade frequency vs 4h.

KEY INSIGHT: Previous failures had either 0 trades (too strict) or 500+ trades
(too loose). This version uses CONSECUTIVE BAR CONFIRMATION (2+ bars) to ensure
breakouts are real, while still generating enough trades (75-150 total).

TIMEFRAME: 12h primary
HTF: 1w for regime, 1d for ATR stop
TARGET: 75-150 total trades over 4 years
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_consec_1w_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    mid = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for regime (bull/bear/range)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Count consecutive bars above upper / below lower (momentum confirmation)
    above_upper_count = np.zeros(n, dtype=np.int32)
    below_lower_count = np.zeros(n, dtype=np.int32)
    
    for i in range(20, n):
        if close[i] > donch_upper[i]:
            above_upper_count[i] = above_upper_count[i-1] + 1 if i > 20 else 1
        if close[i] < donch_lower[i]:
            below_lower_count[i] = below_lower_count[i-1] + 1 if i > 20 else 1
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # 1w regime check
        price_above_1w_hma = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.2
        
        # Consecutive bar momentum (need 2+ bars for confirmation)
        above_count = above_upper_count[i]
        below_count = below_lower_count[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # 2+ consecutive closes above upper band + volume + bullish regime
            if above_count >= 2 and vol_spike and price_above_1w_hma:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # 2+ consecutive closes below lower band + volume + bearish regime
            if below_count >= 2 and vol_spike and not price_above_1w_hma:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
        
        # === EXIT: Return to channel or regime change ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price returns below lower band OR regime turns bearish
            if below_count >= 1:
                exit_triggered = True
            # Also exit if 1w HMA flips bearish while we're long
            if not price_above_1w_hma and above_count == 0:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price returns above upper band OR regime turns bullish
            if above_count >= 1:
                exit_triggered = True
            # Also exit if 1w HMA flips bullish while we're short
            if price_above_1w_hma and below_count == 0:
                exit_triggered = True
        
        if exit_triggered:
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