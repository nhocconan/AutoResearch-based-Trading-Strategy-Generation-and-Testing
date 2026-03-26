#!/usr/bin/env python3
"""
Experiment #009: 4h Donchian Breakout + Volume Spike + 1d HMA Trend

HYPOTHESIS: Donchian channels capture institutional breakout points naturally.
The 20-period (5-day) 4h channel identifies when price breaks out of congestion.
Volume spike confirms "smart money" involvement. 1d HMA filters entries to trend direction.
This works in BOTH bull and bear because Donchian adapts to ANY market structure:
- Bull: long breakouts, trail stop higher
- Bear: short breakouts, trail stop lower
- Range: no breakouts = no trades (correct behavior!)

DB REFERENCE: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (SOL: test_sharpe=1.382, 95tr)
KEY: Keep it SIMPLE. The working DB strategies use 2-3 conditions max.

TARGET: 75-150 total trades over 4 years (19-37/year on 4h)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_hma_simple_v1"
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
    """Average True Range with proper warmup"""
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
    
    # === LOAD 1d DATA ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === CALCULATE 4h INDICATORS ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (current vs 20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Donchian channels (20-period)
    donchian_period = 20
    upper_donch = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_donch = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    bars_since_entry = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # ATR needs ~14 bars, Donchian needs 20, HMA needs ~21*2
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Skip if 1d HMA not ready
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Skip if Donchian not ready
        if np.isnan(upper_donch[i]) or np.isnan(lower_donch[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === EXIT LOGIC ===
        if in_position:
            bars_since_entry += 1
            
            # Update highest/lowest
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
            
            # Check stoploss (2x ATR)
            stop_hit = False
            if position_side > 0 and low[i] < (highest_since_entry - 2.0 * entry_atr):
                stop_hit = True
            if position_side < 0 and high[i] > (lowest_since_entry + 2.0 * entry_atr):
                stop_hit = True
            
            # Check take profit at opposite Donchian
            tp_hit = False
            if position_side > 0:
                # TP when price reaches upper Donchian area
                if high[i] >= upper_donch[i] * 0.998:  # within 0.2% of upper band
                    tp_hit = True
            if position_side < 0:
                # TP when price reaches lower Donchian area
                if low[i] <= lower_donch[i] * 1.002:  # within 0.2% of lower band
                    tp_hit = True
            
            if stop_hit or tp_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                bars_since_entry = 0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC ===
        if not in_position:
            # TREND DIRECTION from 1d HMA
            trend_up = close[i] > hma_1d_aligned[i]
            trend_down = close[i] < hma_1d_aligned[i]
            
            # VOLUME CONFIRMATION
            vol_spike = vol_ratio[i] > 1.5
            
            # DONCHIAN BREAKOUT / PULLBACK
            # Long: price at or slightly below upper Donchian (pullback to breakout level)
            # Short: price at or slightly above lower Donchian (pullback to breakdown level)
            
            dist_to_upper = (upper_donch[i] - close[i]) / atr_14[i] if atr_14[i] > 0 else 999
            dist_to_lower = (close[i] - lower_donch[i]) / atr_14[i] if atr_14[i] > 0 else 999
            
            # At upper band (within 0.3 ATR) for longs
            at_upper = dist_to_upper >= 0 and dist_to_upper <= 0.3
            # At lower band (within 0.3 ATR) for shorts
            at_lower = dist_to_lower >= 0 and dist_to_lower <= 0.3
            
            # ENTRY SIGNAL
            entry_signal = 0.0
            
            # Long: at upper Donchian + volume spike + uptrend
            if at_upper and vol_spike and trend_up:
                entry_signal = SIZE
            
            # Short: at lower Donchian + volume spike + downtrend
            if at_lower and vol_spike and trend_down:
                entry_signal = -SIZE
            
            if entry_signal != 0.0:
                in_position = True
                position_side = int(np.sign(entry_signal))
                entry_atr = atr_14[i]
                bars_since_entry = 0
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = entry_signal
            else:
                signals[i] = 0.0
        else:
            # Maintain position
            signals[i] = SIZE if position_side > 0 else -SIZE
    
    return signals