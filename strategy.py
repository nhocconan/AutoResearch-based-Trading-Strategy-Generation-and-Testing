#!/usr/bin/env python3
"""
Experiment #023: 4h Donchian + ATR Volatility Regime + Volume

HYPOTHESIS: ATR volatility regime (short-term vs long-term ATR ratio) captures
market expansion vs compression cycles. This works in:
- 2021 bull: ATR expands → trending regime activates → catch breakouts
- 2022 bear: ATR expands during crashes → short signals
- 2025 range: ATR compresses → regime filter keeps flat

KEY INSIGHT: ATR(14)/ATR(100) ratio > 0.5 means short-term vol is rising relative
to long-term = market transitioning to trending. This is a cleaner regime filter
than ADX or choppiness because ATR is already calculated for stops.

TRADE COUNT: Target 100-200 total over 4 years (25-50/year).
Size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_atr_regime_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - uses PREVIOUS bars for upper/lower (shifted by 1)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().shift(1).values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().shift(1).values
    middle = (upper + lower) / 2.0
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d_21 = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_21)
    
    # === 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_100 = calculate_atr(high, low, close, period=100)
    upper, middle, lower = calculate_donchian(high, low, period=20)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Need ATR(100) to stabilize
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_100[i]) or atr_100[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === ATR VOLATILITY REGIME ===
        # Ratio > 0.5 means short-term vol is rising = trending regime
        # Ratio < 0.5 means compression = chop (skip trades)
        atr_ratio = atr_14[i] / atr_100[i] if atr_100[i] > 1e-10 else 0
        trending_regime = atr_ratio > 0.5
        
        # === HTF MACRO FILTER ===
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === ENTRY SIGNALS ===
        if not in_position:
            # LONG: Close breaks above upper Donchian + vol spike + HTF bullish + trending
            if close[i] > upper[i] and vol_spike and htf_bullish and trending_regime:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Close breaks below lower Donchian + vol spike + HTF bearish + trending
            elif close[i] < lower[i] and vol_spike and htf_bearish and trending_regime:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
        
        # === EXIT: ATR trailing stop ===
        elif in_position:
            # Minimum hold: 3 bars (12h) to avoid immediate reversals
            min_hold = (i - entry_bar) >= 3
            
            # Check for ATR trailing stop
            if position_side > 0:
                stop_price = highest_since_entry - 2.5 * entry_atr
                if low[i] < stop_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                else:
                    signals[i] = SIZE
            else:
                stop_price = lowest_since_entry + 2.5 * entry_atr
                if high[i] > stop_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                else:
                    signals[i] = -SIZE
    
    return signals