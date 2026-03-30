#!/usr/bin/env python3
"""
Experiment #024: 4h TRIX + Choppiness Regime + Donchian Confirmation

HYPOTHESIS: TRIX (triple-smoothed momentum) catches trend reversals earlier
and more reliably than HMA or single-smoothed indicators. Combined with:
1. Choppiness Index regime filter (avoid trades in range-bound markets)
2. Donchian(20) confirmation (price must break structure)
3. Volume spike confirmation (institutional participation)
4. 1d SMA trend filter (macro direction alignment)

This should work in both:
- 2021 bull: TRIX cross up + break above Donchian + 1d uptrend
- 2022 bear: TRIX cross down + break below Donchian + 1d downtrend
- 2025 range: Choppiness filter keeps us flat when no trend

TARGET: 75-200 trades over 4 years (19-50/year). Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_chop_donchian_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(close, period=15):
    """
    TRIX: Triple-smoothed rate of change.
    More noise-resistant than RSI, catches reversals earlier than moving averages.
    """
    n = len(close)
    if n < period + 3:
        return np.full(n, np.nan)
    
    # Triple EMA smoothing
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Rate of change of triple EMA
    trix = np.zeros(n)
    trix[period*3:] = (ema3.values[period*3:] - ema3.values[period*3 - period:]) / ema3.values[period*3 - period:] * 100
    
    return trix

def calculate_choppiness_index(high, low, close, period=14):
    """
    CHOP < 38.2 = trending market (good for momentum strategies)
    CHOP > 61.8 = ranging market (avoid mean reversion)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    chop = np.full(n, 50.0)
    
    for i in range(period, n):
        highest_high = high[i - period + 1:i + 1].max()
        lowest_low = low[i - period + 1:i + 1].min()
        
        if highest_high - lowest_low > 1e-10:
            sum_tr = 0.0
            for j in range(i - period + 1, i + 1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
                sum_tr += tr
            
            chop[i] = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian channel"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    middle = (upper + pd.Series(low).rolling(window=period, min_periods=period).min().values) / 2
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, middle, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range for stops"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d_50 = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    # === 4h indicators ===
    trix = calculate_trix(close, period=15)
    chop = calculate_choppiness_index(high, low, close, period=14)
    dc_upper, dc_middle, dc_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Pre-compute TRIX crosses ===
    trix_cross_up = np.zeros(n, dtype=bool)
    trix_cross_down = np.zeros(n, dtype=bool)
    prev_trix = np.full(n, np.nan)
    
    for i in range(1, n):
        if not np.isnan(trix[i]) and not np.isnan(trix[i-1]):
            if trix[i-1] <= 0 and trix[i] > 0:
                trix_cross_up[i] = True
            elif trix[i-1] >= 0 and trix[i] < 0:
                trix_cross_down[i] = True
            prev_trix[i] = trix[i-1]
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 80  # Need enough for TRIX triple smoothing
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(trix[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER: Choppiness ===
        # CHOP > 61.8 = ranging, avoid trading
        # CHOP < 50 = trending, allow momentum trades
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 50.0
        
        # === HTF TREND ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === DONCHIAN CONFIRMATION ===
        # Price broke above upper band = bullish momentum confirmed
        # Price broke below lower band = bearish momentum confirmed
        donchian_bull = close[i] > dc_upper[i] if not np.isnan(dc_upper[i]) else False
        donchian_bear = close[i] < dc_lower[i] if not np.isnan(dc_lower[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === MINIMUM HOLD: 3 bars (12h) to avoid chop ===
        min_hold = (i - entry_bar) >= 3
        
        # === Update highest/lowest for trailing stop ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        if in_position:
            if position_side > 0:
                stop_price = highest_since_entry - 2.5 * entry_atr
                if low[i] < stop_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    continue
            else:
                stop_price = lowest_since_entry + 2.5 * entry_atr
                if high[i] > stop_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    continue
            
            # Exit on trend reversal with hold period
            if position_side > 0 and trix_cross_down[i] and min_hold:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                continue
            if position_side < 0 and trix_cross_up[i] and min_hold:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                continue
            
            # Exit on HTF trend reversal
            if position_side > 0 and htf_bearish and min_hold:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                continue
            if position_side < 0 and htf_bullish and min_hold:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                continue
            
            # Hold signal
            signals[i] = position_side * SIZE
            continue
        
        # === NEW POSITIONS ===
        # LONG: TRIX cross up + Donchian break + volume + 1d uptrend
        if trix_cross_up[i] and donchian_bull and htf_bullish:
            in_position = True
            position_side = 1
            entry_atr = atr_14[i]
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        
        # LONG WEAKER: TRIX cross up + 1d uptrend + volume spike (no Donchian break)
        elif trix_cross_up[i] and htf_bullish and vol_spike and not is_choppy:
            in_position = True
            position_side = 1
            entry_atr = atr_14[i]
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE * 0.7  # Smaller size without Donchian confirm
    
    return signals