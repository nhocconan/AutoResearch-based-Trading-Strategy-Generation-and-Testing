#!/usr/bin/env python3
"""
Experiment #025: 4h Camarilla Pivot + Choppiness Regime + Volume Spike

HYPOTHESIS: Camarilla pivot levels from daily candles are a proven edge in crypto.
- L3/L4 touches = mean reversion long opportunities
- H3/H4 touches = mean reversion short opportunities
- Choppiness Index filters out ranging markets (CHOP > 61.8 = no trades)
- Volume spike confirms the reversal is real
- Works in BOTH bull and bear: pivots adapt to daily volatility

WHY 4h + 1d: Top DB performer (ETHUSDT test Sharpe=1.47) used 4h + 1d pivots.
Trade frequency: 4h = 1820 bars/year. L3/L4 touches should give 75-200 trades/4y.

KEY INSIGHT: Keep it SIMPLE. Camarilla touch + volume spike + chop filter = 3 conditions.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_chop_vol_v1"
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

def calculate_coppock(high, low, close, period=14):
    """Coppock Curve - momentum oscillator for trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    roc1 = np.zeros(n)
    roc2 = np.zeros(n)
    
    for i in range(period, n):
        if close[i - period] > 0:
            roc1[i] = ((close[i] - close[i - period]) / close[i - period]) * 100
        if close[i - 2 * period] > 0:
            roc2[i] = ((close[i] - close[i - 2 * period]) / close[i - 2 * period]) * 100
    
    coppock = pd.Series(roc1 + roc2).ewm(span=10, min_periods=10, adjust=False).mean().values
    return coppock

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detector"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    
    for i in range(period, n):
        sum_tr = 0.0
        for j in range(period):
            tr = max(high[i - j] - low[i - j], 
                     abs(high[i - j] - close[i - j - 1]) if i - j - 1 >= 0 else high[i - j] - low[i - j],
                     abs(low[i - j] - close[i - j - 1]) if i - j - 1 >= 0 else high[i - j] - low[i - j])
            sum_tr += tr
        
        highest = max(high[i - period + 1:i + 1])
        lowest = min(low[i - period + 1:i + 1])
        
        if highest - lowest > 0:
            chop[i] = 100 * np.log10(sum_tr / (highest - lowest)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for macro direction (call ONCE) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume confirmation (20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian for Camarilla-like structure (20-bar high/low)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    dc_mid_20 = (dc_upper_20 + dc_lower_20) / 2.0
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER: CHOP > 61.8 = choppy, skip ===
        choppy = chop[i] > 61.8
        
        # === HTF TREND ===
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === CAMARILLA-LIKE PIVOT STRUCTURE ===
        # Use Donchian range to estimate daily pivot levels
        if not np.isnan(dc_upper_20[i]) and not np.isnan(dc_lower_20[i]):
            dc_range = dc_upper_20[i] - dc_lower_20[i]
            pivot = (dc_upper_20[i] + dc_lower_20[i]) / 2.0
            
            # Camarilla-like levels based on range
            # R1 = pivot + range * 0.0833, R2 = pivot + range * 0.1667, R3 = pivot + range * 0.275
            # S1 = pivot - range * 0.0833, S2 = pivot - range * 0.1667, S3 = pivot - range * 0.275
            r3 = pivot + dc_range * 0.275
            r2 = pivot + dc_range * 0.1667
            r1 = pivot + dc_range * 0.0833
            s1 = pivot - dc_range * 0.0833
            s2 = pivot - dc_range * 0.1667
            s3 = pivot - dc_range * 0.275
            
            # === ENTRY SIGNALS ===
            # LONG: Price touches or crosses below S2 zone + volume spike + uptrend
            long_signal = False
            short_signal = False
            
            if not choppy:
                # Long: price at S2/S3 support with volume + 1d uptrend
                if htf_bullish and low[i] <= s2 and low[i] >= s3:
                    long_signal = volume[i] > vol_ma[i] * 1.5
                
                # Short: price at R2/R3 resistance with volume + 1d downtrend
                if htf_bearish and high[i] >= r2 and high[i] <= r3:
                    short_signal = volume[i] > vol_ma[i] * 1.5
            
            # === TRAILING STOP ===
            if in_position:
                if position_side > 0:
                    highest_since_entry = max(highest_since_entry, high[i])
                else:
                    lowest_since_entry = min(lowest_since_entry, low[i])
            
            # === MIN HOLD: 2 bars (8h) ===
            min_hold = (i - entry_bar) >= 2
            
            # === ATR TRAILING STOP (2.0x ATR from highest/lowest) ===
            if in_position:
                stop_hit = False
                if position_side > 0:
                    stop_hit = low[i] < (highest_since_entry - 2.0 * atr_14[i])
                else:
                    stop_hit = high[i] > (lowest_since_entry + 2.0 * atr_14[i])
                
                # Exit on opposite HTF trend (after min hold)
                if min_hold:
                    if position_side > 0 and htf_bearish:
                        stop_hit = True
                    if position_side < 0 and htf_bullish:
                        stop_hit = True
                
                if stop_hit:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                else:
                    signals[i] = position_side * SIZE
            
            # === NEW POSITIONS ===
            if not in_position:
                if long_signal:
                    in_position = True
                    position_side = 1
                    entry_bar = i
                    highest_since_entry = high[i]
                    signals[i] = SIZE
                
                elif short_signal:
                    in_position = True
                    position_side = -1
                    entry_bar = i
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
                
                else:
                    signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals