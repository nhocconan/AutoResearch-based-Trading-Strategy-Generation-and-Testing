#!/usr/bin/env python3
"""
Experiment #028: 4h Williams %R + Donchian Breakout + Choppiness Regime

HYPOTHESIS: Williams %R measures momentum within the recent range. Using extreme
readings (<-80 for longs, >-20 for shorts) ensures we only enter when momentum is
stretched. Combining with actual Donchian(20) CLOSE breakouts (not just touches)
filters out false breakouts. 1d SMA200 confirms trend. Choppiness keeps us out
of range-bound markets.

WHY IT WORKS IN BULL AND BEAR: Uses symmetrical Williams %R extremes, so oversold
shorts work in bear markets, overbought longs work in bull. Donchian confirms
institutional breakout participation. Fewer trades = less fee drag.

TARGET: 75-200 total trades over 4 years (19-50/year). HARD MAX: 300.
Signal size: 0.25 (conservative).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_williams_r_donchian_chop_v2"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        period_high = np.max(high[i - period + 1:i + 1])
        period_low = np.min(low[i - period + 1:i + 1])
        range_hl = period_high - period_low
        
        if range_hl > 0:
            willr[i] = -100 * (period_high - close[i]) / range_hl
    
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
    
    # 1d SMA200 for trend direction
    sma_200 = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    willr = calculate_williams_r(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels (20 periods = 3.3 days on 4h)
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
    
    warmup = 220  # Need enough for SMA200(1d) + Donchian(20) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
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
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_200_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # Only trade in trending or neutral markets
        # CHOP > 61.8 = too choppy, skip
        is_choppy = chop[i] > 61.8
        
        # Skip if too choppy (only when flat)
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # === MOMENTUM (Williams %R) ===
        # Oversold (<-80) = potential long setup
        # Overbought (>-20) = potential short setup
        willr_oversold = willr[i] < -80
        willr_overbought = willr[i] > -20
        
        # === DONCHIAN BREAKOUT (require CLOSE to break, not just touch) ===
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        prev_close = close[i - 1] if i > 0 else close[i]
        
        # Breakout: previous bar closed below, this bar closes above (or vice versa)
        broke_high = prev_close < prev_donchian_high and close[i] > prev_donchian_high
        broke_low = prev_close > prev_donchian_low and close[i] < prev_donchian_low
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Oversold + Breakout above ===
            # Williams %R < -80 (oversold) + price closes above Donchian high
            if broke_high and willr_oversold and price_above_1d_sma:
                if vol_spike:  # Volume confirmation required
                    desired_signal = SIZE
            
            # === SHORT: Overbought + Breakdown below ===
            # Williams %R > -20 (overbought) + price closes below Donchian low
            if broke_low and willr_overbought and not price_above_1d_sma:
                if vol_spike:  # Volume confirmation required
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLDING PERIOD (minimum 8 bars = 2 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit on Williams %R reversal
            if position_side > 0 and willr[i] > -20:  # No longer oversold
                desired_signal = 0.0
            if position_side < 0 and willr[i] < -80:  # No longer overbought
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