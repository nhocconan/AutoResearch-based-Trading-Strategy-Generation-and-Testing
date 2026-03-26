#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + Volume + 1d Trend

HYPOTHESIS: Donchian(20) breakout marks institutional entry points.
Volume spike (>1.5x MA20) confirms the move is real.
1d HMA(21) alignment filters trades against the major trend.
ATR(14) stoploss at 2.5x controls risk.
4h is fast enough for meaningful trades (75-125 over 4yr) but slow enough
to avoid overtrading. Works in both bull (buy breakouts above 1d HMA)
and bear (sell breakouts below 1d HMA, including breakdown shorts).

Entry: Breakout OR price outside channel + volume spike + 1d trend aligned
Exit: Opposite band break OR 3xATR stoploss
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_1d_hma_v1"
timeframe = "4h"
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
    """Donchian Channel - returns upper, lower, and midpoint"""
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
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend alignment
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Volume MA20 and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size - conservative but meaningful
    
    # Position state
    in_position = False
    position_side = 0  # 1=long, -1=short
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Guard: check all indicators ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === CONDITION PRECOMPUTATION ===
        # 1d trend alignment
        bullish_trend = close[i] > hma_1d_aligned[i]
        bearish_trend = close[i] < hma_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian state
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        price_above_mid = close[i] > donch_mid[i]
        price_below_mid = close[i] < donch_mid[i]
        
        # ATR stop level
        stop_mult = 2.5
        long_stop = close[i] - stop_mult * atr_14[i]
        short_stop = close[i] + stop_mult * atr_14[i]
        
        desired_signal = 0.0
        
        # === NEW ENTRY LOGIC ===
        if not in_position:
            # LONG: Breakout above upper band + volume + bullish 1d
            if price_above_upper and vol_spike and bullish_trend:
                desired_signal = SIZE
            
            # SHORT: Breakdown below lower band + volume + bearish 1d
            if price_below_lower and vol_spike and bearish_trend:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        stoploss_hit = False
        exit_signal = False
        
        if in_position:
            # Update highest/lowest
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
            
            # ATR stoploss check
            if position_side > 0:
                if low[i] < (entry_price - stop_mult * entry_atr):
                    stoploss_hit = True
            else:
                if high[i] > (entry_price + stop_mult * entry_atr):
                    stoploss_hit = True
            
            # Opposite band exit (price returns to channel)
            if position_side > 0 and price_below_lower:
                exit_signal = True
            if position_side < 0 and price_above_upper:
                exit_signal = True
            
            if stoploss_hit or exit_signal:
                desired_signal = 0.0
        
        # === POSITION MANAGEMENT ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        elif in_position:
            # Exit
            in_position = False
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals