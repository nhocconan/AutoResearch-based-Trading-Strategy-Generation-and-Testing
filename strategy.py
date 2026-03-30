#!/usr/bin/env python3
"""
Experiment #026: 4h Donchian(20) + Choppiness Regime + Volume Spike

HYPOTHESIS: Choppiness Index (CHOP) is the optimal regime filter from 16K+ experiments.
It distinguishes trending vs ranging markets better than ADX or EMA alone.
- CHOP < 38.2 = trending market → follow breakouts
- CHOP > 61.8 = ranging market → avoid (too many false breakouts)

WHY IT WORKS IN BULL AND BEAR:
- Bull (2020-2021, 2024-2025): CHOP<38 + Donchian breakout = strong uptrend continuation
- Bear (2022): CHOP>61 = stay out of chop, wait for clear breakouts
- Range: CHOP>61 means skip → avoids 2022 whipsaw which destroys trend strategies

ENTRY: CHOP < 40 + Close > Donchian High(20) + Volume > 1.5x MA(20)
SHORT: CHOP < 40 + Close < Donchian Low(20) + Volume > 1.5x MA(20)
EXIT: Opposite signal or ATR 2.5x trailing stop

TARGET: 80-150 total over 4 years (20-37/year). Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_chop_volume_v3"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of true range over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
        
        # Highest high - lowest low over period
        highest_high = max(high[i - period + 1:i + 1])
        lowest_low = min(low[i - period + 1:i + 1])
        hl_range = highest_high - lowest_low
        
        if hl_range > 1e-10:
            # CHOP = 100 * log10(sum of TR / HL range) / log10(N)
            chop[i] = 100 * np.log10(tr_sum / hl_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA for trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=30, min_periods=30).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === 4h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Donchian 20 (shift by 1 to avoid look-ahead)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume MA(20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        
        if np.isnan(chop_14[i]) or np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME CHECK: Only trade in trending market ===
        is_trending = chop_14[i] < 40.0  # Stricter threshold for fewer trades
        
        # === TREND DIRECTION FROM 1d SMA ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        bullish_breakout = (close[i] > dc_upper_20[i]) if not np.isnan(dc_upper_20[i]) else False
        bearish_breakout = (close[i] < dc_lower_20[i]) if not np.isnan(dc_lower_20[i]) else False
        
        # === VOLUME CONFIRMATION (stronger: 1.5x) ===
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False
        
        # === TRAILING STOP UPDATE ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === MIN HOLD: 3 bars (12h) to avoid immediate whipsaw ===
        min_hold = (i - entry_bar) >= 3
        
        # === STOPLOSS CHECK (ATR trailing) ===
        stop_hit = False
        if in_position:
            if position_side > 0:
                stop_hit = low[i] < (highest_since_entry - 2.5 * atr_14[i])
            else:
                stop_hit = high[i] > (lowest_since_entry + 2.5 * atr_14[i])
            
            # Exit on trend reversal (after min hold)
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
            continue
        
        # === NEW POSITIONS (only in trending regime) ===
        # Long: Trending market + bullish breakout + volume confirm
        if is_trending and bullish_breakout and vol_ok and htf_bullish:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        
        # Short: Trending market + bearish breakdown + volume confirm
        elif is_trending and bearish_breakout and vol_ok and htf_bearish:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals