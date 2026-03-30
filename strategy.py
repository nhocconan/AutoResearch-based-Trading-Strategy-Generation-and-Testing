#!/usr/bin/env python3
"""
Experiment #028: 6h Donchian Breakout + Williams %R Momentum + Choppiness + 1d SMA50

HYPOTHESIS: Donchian breakout alone is too loose. Adding Williams %R momentum
confirmation (< 20 for longs, > 80 for shorts) filters false breakouts where
price breaks out but immediately reverts. Williams %R < 20 means price closed
in top 20% of 14-period range = genuine momentum.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Long breakouts with Williams %R < 20 = strong upward momentum
- Bear: Short breakouts with Williams %R > 80 = strong downward momentum
- Choppiness filter < 55 keeps us out of range-bound markets
- 1d SMA50 ensures we don't fight major trends

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 200.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_williams_chop_1d_v1"
timeframe = "6h"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(high)
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    willr = calculate_williams_r(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels (20 periods = 5 days on 6h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough for Donchian(20) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(willr[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # Only trade when not too choppy (CHOP < 55 = trending or mild chop)
        # This is tighter than 61.8 to reduce whipsaws
        is_not_choppy = chop[i] < 55.0
        
        # Skip if too choppy
        if is_not_choppy == False and not in_position:
            signals[i] = 0.0
            continue
        
        # === WILLIAMS %R MOMENTUM ===
        # < -20 means price closed near top of range = bullish momentum
        # > -80 means price closed near bottom of range = bearish momentum
        strong_bullish = willr[i] < -80  # Very oversold, strong upward momentum
        strong_bearish = willr[i] > -20  # Very overbought, strong downward momentum
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        current_high = high[i]
        current_low = low[i]
        
        # Previous bar's Donchian values (shift by 1 to avoid look-ahead)
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        # Volume confirmation (optional - adds one more filter)
        vol_spike = vol_ratio[i] > 2.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high + Williams %R momentum ===
            # All 4 conditions must align:
            # 1. Price above 1d SMA (trend)
            # 2. Not too choppy (CHOP < 55)
            # 3. Williams %R < -80 (momentum)
            # 4. Break above previous Donchian high
            if current_high > prev_donchian_high and price_above_1d_sma:
                if strong_bullish:  # Williams %R confirms momentum
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low + Williams %R momentum ===
            if current_low < prev_donchian_low and not price_above_1d_sma:
                if strong_bearish:  # Williams %R confirms momentum
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 8 bars = 2 days on 6h) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit if Williams %R reverts to neutral
            if position_side > 0 and willr[i] > -30:  # Momentum fading
                desired_signal = 0.0
            if position_side < 0 and willr[i] < -70:  # Price recovering
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
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals