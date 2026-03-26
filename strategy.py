#!/usr/bin/env python3
"""
Experiment #003: 4h HMA Trend + Donchian Breakout + Williams %R + Volume

HYPOTHESIS: Institutional breakout detection on 4h TF using Donchian(20) channels.
Entry requires:
1. 12h HMA trend alignment (direction filter - reduces false breakouts)
2. Williams %R momentum confirmation (<20 for longs, >80 for shorts)
3. Volume spike >1.5x average (institutional participation required)
4. Donchian channel breakout (price structure confirmation)

This combination worked in DB (SOLUSDT 1.38 Sharpe, 95 trades). The 12h HTF
trend filter reduces whipsaws while keeping trade frequency manageable.

TIMEFRAME: 4h primary, 12h for trend
TARGET: 75-200 total trades over 4 years (tight entry = fewer trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_williams_vol_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - precomputed for speed"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    # Use pandas for rolling WMA (faster than pure Python)
    wma_half = pd.Series(close).rolling(half, min_periods=half).mean().values
    wma_full = pd.Series(close).rolling(period, min_periods=period).mean().values
    
    # diff = 2*WMA(half) - WMA(full)
    diff = np.where(
        ~np.isnan(wma_half) & ~np.isnan(wma_full),
        2.0 * wma_half - wma_full,
        np.nan
    )
    
    # HMA = WMA(sqrt(period)) of diff
    hma = pd.Series(diff).rolling(sqrt_n, min_periods=sqrt_n).mean().values
    
    return hma

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper and lower bands"""
    n = len(high)
    upper = pd.Series(high).rolling(period, min_periods=period).max().values
    lower = pd.Series(low).rolling(period, min_periods=period).min().values
    return upper, lower

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    highest = pd.Series(high).rolling(period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(period, min_periods=period).min().values
    
    wr = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(highest[i]) and not np.isnan(lowest[i]):
            range_val = highest[i] - lowest[i]
            if range_val > 0:
                wr[i] = -100 * (highest[i] - close[i]) / range_val
    
    return wr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h HMA for trend (must align to 4h bars with shift(1))
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # 12h HMA slope for additional confirmation
    hma_12h_slope = pd.Series(hma_12h_aligned).diff(1).values
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    williams_r = calculate_williams_r(high, low, close, period=14)
    
    # Volume MA and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Donchian mid channel
    donch_mid = (donch_upper + donch_lower) / 2
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if critical indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === HTF TREND (12h HMA) ===
        # Bullish: price above 12h HMA
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        # HMA rising = confirming bullish
        hma_rising = hma_12h_slope[i] > 0 if not np.isnan(hma_12h_slope[i]) else False
        
        # === LOCAL MOMENTUM (Williams %R) ===
        wr_val = williams_r[i] if not np.isnan(williams_r[i]) else -50.0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN STRUCTURE ===
        price_near_upper = close[i] > donch_mid[i]  # Above midpoint = bullish structure
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # 1. Price above 12h HMA (trend aligned)
            # 2. Williams %R < -80 (oversold, momentum turning up)
            # 3. Volume spike >1.5x
            # 4. Price above Donchian midpoint (structure confirms)
            if (price_above_hma_12h and 
                wr_val < -80 and 
                vol_spike and 
                price_near_upper):
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # 1. Price below 12h HMA (trend aligned)
            # 2. Williams %R > -20 (overbought, momentum turning down)
            # 3. Volume spike >1.5x
            # 4. Price below Donchian midpoint (structure confirms)
            if (not price_above_hma_12h and 
                wr_val > -20 and 
                vol_spike and 
                not price_near_upper):
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (prevent holding forever) ===
        bars_since_entry = i - (i - sum(signals[:i] != 0)) if position_side != 0 else 0
        # Simple: exit if Williams %R reaches opposite extreme
        if in_position and position_side > 0 and wr_val > -20:
            desired_signal = 0.0  # Take profit on overbought
        if in_position and position_side < 0 and wr_val < -80:
            desired_signal = 0.0  # Take profit on oversold
        
        # === OPPOSITE SIGNAL EXIT ===
        # If we get a strong opposite signal, exit current position
        if in_position and position_side > 0:
            # Check for short signal
            if (not price_above_hma_12h and 
                wr_val > -20 and 
                vol_ratio[i] > 1.5 and 
                not price_near_upper):
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Check for long signal
            if (price_above_hma_12h and 
                wr_val < -80 and 
                vol_ratio[i] > 1.5 and 
                price_near_upper):
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals