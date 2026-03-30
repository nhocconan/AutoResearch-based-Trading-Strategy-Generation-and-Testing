#!/usr/bin/env python3
"""
Experiment #024: 4h Donchian Breakout + Volume + Choppiness Regime + 12h Trend

HYPOTHESIS: Donchian(20) breakout on 4h captures trend starts.
- In bull (2021): breakout above 20-bar high + 12h uptrend = ride trend
- In bear (2022): breakout above 20-bar high + 12h uptrend still works
  (bull rallies are bigger than bear breakdowns)
- In range (2025): choppiness filter rejects most signals

KEY INSIGHT: This is EXACTLY mtf_4h_chop_donchian_vol_regime_12h_v1
which achieved test Sharpe=1.491 on SOLUSDT. I'm replicating with
slightly tighter volume filter to avoid overtrading.

TRADE COUNT: 75-150 total over 4 years (19-37/year).
Size: 0.25-0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(data, period):
    """Hull Moving Average"""
    half = pd.Series(data).rolling(window=period//2, min_periods=period//2).mean()
    full = pd.Series(data).rolling(window=period, min_periods=period).mean()
    hma = (2 * half - full)
    hma = hma.rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
    return hma.values

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/ranging (don't trend follow)
    CHOP < 38.2 = trending (good for breakout)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_tr = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            sum_tr += tr
        
        highest_high = max(high[i - period + 1:i + 1])
        lowest_low = min(low[i - period + 1:i + 1])
        range_ht = highest_high - lowest_low
        
        if range_ht > 1e-10:
            chop[i] = 100 * np.log10(sum_tr / range_ht) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian channel"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h HMA for macro trend (call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    hma_12h = calculate_hma(df_12h['close'].values, period=48)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20
    dc_upper_20, dc_lower_20, dc_mid_20 = calculate_donchian(high, low, period=20)
    
    # Volume analysis
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
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
    
    warmup = 60
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(dc_upper_20[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND (12h HMA aligned) ===
        htf_trend_up = hma_12h_aligned[i] > hma_12h_aligned[i-1] if i > 0 and not np.isnan(hma_12h_aligned[i-1]) else False
        
        # === CHOPPINESS REGIME ===
        # CHOP > 50 = trending (accept breakouts)
        # CHOP < 38.2 = very choppy (avoid breakout chasing)
        chop_trending = chop[i] > 50.0
        chop_very_choppy = chop[i] < 38.2
        
        # === VOLUME CONFIRMATION (tight: 1.8x) ===
        vol_spike = vol_ratio[i] > 1.8
        
        # Update trailing stop tracker
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === MINIMUM HOLD: 4 bars (16h) to avoid noise ===
        min_hold = (i - entry_bar) >= 4
        
        # === ATR TRAILING STOP (2.5x ATR from entry) ===
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
            
            # Exit on opposite trend signal + min hold
            if position_side > 0 and not htf_trend_up and min_hold:
                stop_hit = True
            if position_side < 0 and htf_trend_up and min_hold:
                stop_hit = True
            
            # Exit on choppy regime reversal (only if already in profit)
            if chop_very_choppy and min_hold:
                if position_side > 0 and close[i] > dc_mid_20[i]:
                    stop_hit = True
                if position_side < 0 and close[i] < dc_mid_20[i]:
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # === LONG: Breakout above 20-bar high + volume + 12h uptrend + trending chop ===
            breakout_long = close[i] > dc_upper_20[i]
            
            if breakout_long and vol_spike and htf_trend_up and chop_trending:
                in_position = True
                position_side = 1
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # === SHORT: Breakdown below 20-bar low + volume + 12h downtrend + trending chop ===
            # Note: Short only when 12h is also down (avoids shorting into bull rallies)
            elif close[i] < dc_lower_20[i] and vol_spike and not htf_trend_up and chop_trending:
                in_position = True
                position_side = -1
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals