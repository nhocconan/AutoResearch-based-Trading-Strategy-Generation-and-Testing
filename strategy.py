#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + Volume Spike + ATR Regime + HMA Direction

HYPOTHESIS: Donchian(20) breakout is a proven institutional price structure signal.
By requiring:
1. Price breaks Donchian(20) channel
2. Volume spike (>1.8x 20-bar MA) — confirms institutional participation
3. ATR(14) above its EMA — confirms trending market (not chop)
4. HMA(16) direction on 12h — confirms trend alignment

This creates TIGHT entry conditions that generate ~75-150 trades/4yr.

WHY IT WORKS IN BULL AND BEAR:
- In bull (2021): HMA up = only long breakouts, short filtered
- In bear (2022): HMA down = only short breakouts, long filtered  
- In recovery (2023-2024): HMA up = long breakouts again

KEY DIFFERENCE FROM FAILED STRATS:
- Previous attempts: loose volume (>1.2x) = too many trades
- This attempt: strict volume (>1.8x) + ATR regime = fewer, higher quality trades

TARGET: 75-150 total over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_atrregime_hma_v1"
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

def calculate_hma(data, period):
    """Hull Moving Average"""
    half_len = period // 2
    sqrt_len = int(np.sqrt(period))
    
    wma_half = pd.Series(data).rolling(window=half_len, min_periods=half_len).mean()
    wma_full = pd.Series(data).rolling(window=period, min_periods=period).mean()
    diff = 2 * wma_half - wma_full
    
    hma = diff.rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h HMA(16) for trend direction
    hma_12h_raw = calculate_hma(df_12h['close'].values, 16)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR regime: trending when ATR > ATR EMA
    atr_ema = pd.Series(atr_14).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_trending = atr_14 > atr_ema
    
    # Donchian(20) channels
    donchian_period = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume spike (>1.8x = strict confirmation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    hold_bars = 0
    
    warmup = 50  # Need enough for indicators
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if HMA not aligned
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (12h HMA) ===
        hma_trending_up = close[i] > hma_12h_aligned[i]
        hma_trending_down = close[i] < hma_12h_aligned[i]
        
        # ATR regime confirmation
        is_trending = atr_trending[i]
        
        # Volume confirmation (strict >1.8x)
        vol_spike = vol_ratio[i] > 1.8
        
        # Donchian breakout signals
        upper_broken = close[i] > donchian_upper[i]
        lower_broken = close[i] < donchian_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Upper Donchian breakout + ALL confirmations ===
            if upper_broken and vol_spike and is_trending and hma_trending_up:
                desired_signal = SIZE
            
            # === SHORT: Lower Donchian breakout + ALL confirmations ===
            if lower_broken and vol_spike and is_trending and hma_trending_down:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT: Exit on Donchian reversal ===
        if in_position:
            hold_bars = i - entry_bar
            
            # Long exit: price falls back below mid-channel
            if position_side > 0 and close[i] < donchian_mid[i] and hold_bars >= 2:
                desired_signal = 0.0
            
            # Short exit: price rises back above mid-channel
            if position_side < 0 and close[i] > donchian_mid[i] and hold_bars >= 2:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                hold_bars = 0
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
                hold_bars = 0
        
        signals[i] = desired_signal
    
    return signals