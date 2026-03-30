#!/usr/bin/env python3
"""
Experiment #004: 1d TRIX Momentum + Choppiness Regime + Volume Spike

HYPOTHESIS: TRIX momentum captures trend acceleration without noise. Choppiness
Index (CHOP) separates trending from ranging markets - only take signals when
CHOP < 38.2 (trending). Combined with volume spike confirmation and 1w SMA for
macro direction, this should work across all market regimes:
- 2021 bull: TRIX crosses up with CHOP trending → long
- 2022 bear: TRIX crosses down with CHOP trending → short
- 2025 range: CHOP > 61.8 → flat (no trades in chop)
- 2025 breakout: CHOP drops below 38.2 + TRIX cross → position

KEY INSIGHT: CHOP is the meta-filter that prevents the biggest failure mode:
taking momentum signals during choppy/ranging markets. DB shows TRIX+chop
achieved ETHUSDT test Sharpe 1.32.

TRADE COUNT: 30-80 total over 4 years (7-20/year on 1d).
Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_trix_chop_vol_1w_sma_v1"
timeframe = "1d"
leverage = 1.0

def calculate_trix(close, period=15):
    """Triple EMA momentum oscillator"""
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # TRIX = rate of change of triple EMA
    trix = np.zeros(len(close))
    trix[0] = 0.0
    for i in range(1, len(close)):
        if ema3.iloc[i-1] != 0:
            trix[i] = ((ema3.iloc[i] - ema3.iloc[i-1]) / ema3.iloc[i-1]) * 100
        else:
            trix[i] = 0.0
    
    # Signal line = SMA of TRIX
    trix_series = pd.Series(trix)
    signal = trix_series.rolling(window=9, min_periods=9).mean().values
    
    return trix, signal

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP): 100 * log10(sum ATR over period) / (max-high - min-low)
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        hl_range = highest_high - lowest_low
        
        if hl_range > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / hl_range) / np.log10(period)
        else:
            chop[i] = 61.8  # default to choppy
    
    return chop

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

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w SMA for macro trend (call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    sma_1w_50 = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_50)
    
    # === 1d Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    trix, trix_signal = calculate_trix(close, period=15)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume metrics
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # State tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Need 1w SMA + TRIX warmup
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(trix[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === CHOPPINESS REGIME FILTER ===
        # Only trade when CHOP < 38.2 (trending), avoid CHOP > 61.8 (choppy)
        chop_trending = chop[i] < 38.2
        chop_choppy = chop[i] > 61.8
        
        # === TRIX MOMENTUM SIGNALS ===
        # TRIX crossing above signal = bullish momentum
        trix_bullish = trix[i] > trix_signal[i]
        # TRIX crossing below signal = bearish momentum
        trix_bearish = trix[i] < trix_signal[i]
        
        # TRIX zero line cross detection (approximation via sign change)
        trix_above_zero = trix[i] > 0
        trix_below_zero = trix[i] < 0
        
        # === 1w MACRO TREND ===
        htf_bullish = close[i] > sma_1w_aligned[i]
        htf_bearish = close[i] < sma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MIN HOLD: 2 bars (2 days) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Exit on momentum reversal with min hold
            if position_side > 0 and trix_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and trix_bullish and min_hold:
                stop_hit = True
            
            # Exit if choppy market (opposite of our thesis)
            if chop_choppy:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # Skip if choppy - no entry in ranging markets
            if chop_choppy:
                signals[i] = 0.0
                continue
            
            # LONG: TRIX bullish + volume spike + CHOP trending + 1w bullish
            # Require TRIX momentum strengthening (cross above signal or above zero)
            if trix_bullish and vol_spike and chop_trending and htf_bullish:
                in_position = True
                position_side = 1
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: TRIX bearish + volume spike + CHOP trending + 1w bearish
            elif trix_bearish and vol_spike and chop_trending and htf_bearish:
                in_position = True
                position_side = -1
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals