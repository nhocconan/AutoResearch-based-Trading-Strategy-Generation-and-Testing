#!/usr/bin/env python3
"""
Experiment #024: 12h Donchian Choppiness Regime

HYPOTHESIS: Donchian(20) breakout on 12h provides clean price structure
signals. Combined with:
- 1d SMA for macro trend (prevents counter-trend trades)
- Choppiness Index for regime detection (only trade when CHOP < 50)
- Volume confirmation (filters false breakouts)
- ATR stoploss (2.5x) for risk management

This is the proven pattern from DB: test Sharpe 1.49, 107 trades on SOL.

Expected: 60-120 total trades over 4 years (15-30/year). Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_chop_vol_v2"
timeframe = "12h"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging market (no trend)
    CHOP < 38.2 = trending market (follow trend)
    Formula: 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else 0)
            tr_sum += tr
        
        # Highest high over period
        high_max = max(high[i - period + 1:i + 1])
        
        # Lowest low over period
        low_min = min(low[i - period + 1:i + 1])
        
        range_hl = high_max - low_min
        
        if range_hl > 1e-10:
            chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian channel - upper = highest high, lower = lowest low"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA 50 for macro trend
    sma_1d_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    # === 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    
    # Donchian(20) - price structure
    dc_upper_20, dc_middle_20, dc_lower_20 = calculate_donchian(high, low, period=20)
    
    # Donchian(10) - faster breakout for confirmation
    dc_upper_10, _, dc_lower_10 = calculate_donchian(high, low, period=10)
    
    # Volume: 20-bar moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signal array ===
    signals = np.zeros(n, dtype=np.float64)
    SIZE = 0.30
    
    # Position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 80  # Need at least 80 bars for 20-bar Donchian + 50 SMA
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop_14[i]) or np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === Update trailing stop tracking ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === Regime filter: CHOP < 50 = trending, ok to trade ===
        is_trending = chop_14[i] < 50.0
        
        # === Macro trend from 1d SMA ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === Volume confirmation: 1.3x average ===
        vol_confirm = vol_ratio[i] >= 1.3
        
        # === Check for new 12h breakout ===
        # Upper breakout: price crosses above 20-bar high
        upper_breakout = close[i] > dc_upper_20[i] if not np.isnan(dc_upper_20[i]) else False
        # Lower breakout: price crosses below 20-bar low
        lower_breakout = close[i] < dc_lower_20[i] if not np.isnan(dc_lower_20[i]) else False
        
        # Confirm with faster Donchian
        upper_confirm = close[i] > dc_upper_10[i] if not np.isnan(dc_upper_10[i]) else False
        lower_confirm = close[i] < dc_lower_10[i] if not np.isnan(dc_lower_10[i]) else False
        
        # === ATR stoploss check ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === MIN HOLD: 2 bars (24h) to avoid whipsaws ===
        min_hold = (i - entry_bar) >= 2
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Trend reversal exits
            if position_side > 0 and htf_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and htf_bullish and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                # Maintain position
                signals[i] = position_side * SIZE
        
        # === NEW ENTRIES ===
        if not in_position:
            # LONG: Upper breakout + trending + volume + 1d uptrend
            if upper_breakout and upper_confirm and is_trending and vol_confirm and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Lower breakout + trending + volume + 1d downtrend
            elif lower_breakout and lower_confirm and is_trending and vol_confirm and htf_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            # No signal
            else:
                signals[i] = 0.0
    
    return signals