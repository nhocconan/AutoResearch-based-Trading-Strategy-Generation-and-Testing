#!/usr/bin/env python3
"""
Experiment #006: 4h ATR Volatility Expansion + Volume + 1d EMA Trend

HYPOTHESIS: Major moves are preceded by volatility COMPRESSION then EXPANSION.
When ATR(5)/ATR(20) crosses above 1.3x, volatility is expanding — the market is
"waking up." Combining this with volume confirmation (>1.5x average) and 1d EMA
trend alignment creates high-probability entries that work in BOTH bull and bear.

WHY ATR REGIME: Unlike fixed thresholds (ADX>25), ATR ratio adapts to each coin's
volatility. BTC has different ATR levels than SOL. This is coin-agnostic.

WHY IT WORKS: Volatility expansion = institutional activity. Volume confirms smart
money. 1d EMA keeps us on the right side of the larger trend.

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_atrregime_vol_ema50_1d_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_5 = calculate_atr(high, low, close, period=5)
    atr_20 = calculate_atr(high, low, close, period=20)
    
    # ATR regime: expansion vs compression
    atr_ratio = atr_5 / np.where(atr_20 > 1e-10, atr_20, np.nan)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_5[i]) or np.isnan(atr_20[i]) or atr_5[i] <= 1e-10 or atr_20[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if ATR ratio not ready
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME DETECTION ===
        atr_expanding = atr_ratio[i] > 1.3  # Volatility is expanding
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === ENTRY CONDITIONS ===
        # Both conditions must be met
        both_confirm = atr_expanding and vol_spike
        
        # === POSITION MANAGEMENT ===
        desired_signal = 0.0
        
        if in_position:
            bars_held = i - entry_bar
            
            # Take profit at 1.5R
            if position_side > 0:
                profit_target = entry_price + 1.5 * entry_atr
                if high[i] >= profit_target:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    signals[i] = desired_signal
                    continue
                    
            elif position_side < 0:
                profit_target = entry_price - 1.5 * entry_atr
                if low[i] <= profit_target:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    signals[i] = desired_signal
                    continue
            
            # Stop loss at 2.0 ATR
            if position_side > 0:
                stop_price = entry_price - 2.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    signals[i] = desired_signal
                    continue
                    
            elif position_side < 0:
                stop_price = entry_price + 2.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    signals[i] = desired_signal
                    continue
            
            # Force exit after 20 bars (80 hours = ~3 days) to avoid holding through chop
            if bars_held >= 20:
                desired_signal = 0.0
                in_position = False
                position_side = 0
                signals[i] = desired_signal
                continue
            
            # Otherwise hold
            desired_signal = position_side * SIZE
        
        else:
            # === NEW ENTRIES ===
            if both_confirm:
                # LONG: price above 1d EMA = bull trend
                if price_above_1d_ema:
                    desired_signal = SIZE
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    entry_atr = atr_20[i]
                    entry_bar = i
                
                # SHORT: price below 1d EMA = bear trend
                elif not price_above_1d_ema:
                    desired_signal = -SIZE
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    entry_atr = atr_20[i]
                    entry_bar = i
        
        signals[i] = desired_signal
    
    return signals