#!/usr/bin/env python3
"""
Experiment #006: 12h Camarilla + Choppiness + Volume Spike

HYPOTHESIS: Camarilla S4/R4 levels are institutional order zones where price
reverses with volume spike confirmation. Choppiness Index filters out
trending markets (where Camarilla fails) and identifies ranging conditions
where mean reversion works.

12h captures multi-day swings while keeping trade count manageable.
Volume spike confirms institutional interest at the level.
HTF EMA21 alignment ensures we're trading WITH the larger trend.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Buy S4 bounces with 1d trend UP = high probability reversal
- Bear: Sell R4 spikes with 1d trend DOWN = catches bear rallies
- Range: Works best (CHOP > 50) when price oscillates at levels

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_chop_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index: <38.2 = trending, >61.8 = choppy/ranging"""
    n = len(close)
    chop = np.full(n, np.nan)
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        if highest == lowest:
            chop[i] = 100.0
        else:
            atr_sum = np.sum(np.abs(close[i-period+1:i+1] - close[i-period:i]))
            range_val = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_val) / np.log10(period)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    ema21_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = pd.Series(atr_14).rolling(30, min_periods=30).mean().values
    atr_ratio = atr_14 / np.where(atr_30 > 0, atr_30, 1)
    
    vol_ma20 = pd.Series(volume).rolling(20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, 1)
    
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        if np.isnan(ema21_1d_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === FILTERS ===
        # HTF trend: price above 1d EMA21 = bullish
        htf_bull = close[i] > ema21_1d_aligned[i]
        htf_bear = close[i] < ema21_1d_aligned[i]
        
        # Volume spike confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # ATR ratio: not in extreme volatility expansion
        atr_normal = atr_ratio[i] < 2.0
        
        # Choppiness: ranging market = good for Camarilla
        is_choppy = chop[i] > 50.0
        
        # === CAMARILLA LEVELS (previous closed bar) ===
        prev_close = close[i - 1]
        prev_range = high[i - 1] - low[i - 1]
        
        r4 = prev_close + prev_range * 0.18333
        s4 = prev_close - prev_range * 0.18333
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: S4 touch + volume + HTF bullish ===
            if htf_bull and vol_spike and is_choppy:
                if low[i] <= s4:
                    desired_signal = SIZE
            
            # === SHORT: R4 touch + volume + HTF bearish ===
            if htf_bear and vol_spike and is_choppy:
                if high[i] >= r4:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD: 2 bars to avoid fee churn ===
        bars_held = i - entry_bar
        if in_position and bars_held < 2:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals