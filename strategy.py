#!/usr/bin/env python3
"""
Experiment #021: 6h Williams %R + 1d Donchian + Trend

HYPOTHESIS: 1d-period Donchian channels (20 calendar days) provide structural
breakout signals on a coarser time frame than typical bar-based channels.
Combined with 6h Williams %R for momentum timing, 1d SMA50 for trend, and
Choppiness Index to avoid range-bound markets, this captures institutional
breakout moves while filtering noise.

WHY 6h + 1d: 1d Donchian (20 days) is wider than 6h (20 bars = 5 days), so fewer
but higher quality signals. 6h Williams %R times entries within the 1d structure.

TARGET: 50-100 total trades over 4 years = 12-25/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williams_donchian_1d_trend_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - values above 61.8 = choppy, below 38.2 = trending"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    wr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest != lowest:
            wr[i] = -100 * (highest - close[i]) / (highest - lowest)
    
    return wr

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
    
    # 1d Donchian 20 (20 calendar days = structural breakouts)
    # Use pre-aligned close from HTF data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_period = 20
    if len(close_1d) >= donchian_period:
        # Upper = highest high over 20d, Lower = lowest low over 20d
        upper_1d = pd.Series(high_1d).rolling(window=donchian_period, min_periods=donchian_period).max().values
        lower_1d = pd.Series(low_1d).rolling(window=donchian_period, min_periods=donchian_period).min().values
        mid_1d = (upper_1d + lower_1d) / 2
        
        # Align to 6h
        upper_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
        lower_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
        mid_aligned = align_htf_to_ltf(prices, df_1d, mid_1d)
    else:
        upper_aligned = np.full(n, np.nan)
        lower_aligned = np.full(n, np.nan)
        mid_aligned = np.full(n, np.nan)
    
    # Local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    williams_r = calculate_williams_r(high, low, close, period=14)
    
    # Smooth Williams %R
    wr_smooth = pd.Series(williams_r).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 200  # SMA50 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === TREND (1d SMA50) ===
        price_above_sma = close[i] > sma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT (1d period) ===
        donch_breakout_up = close[i] > upper_aligned[i] if not np.isnan(upper_aligned[i]) else False
        donch_breakout_down = close[i] < lower_aligned[i] if not np.isnan(lower_aligned[i]) else False
        
        # === REGIME (Choppiness) ===
        # Only trade in trending or neutral market (avoid choppy)
        is_tradeable = chop[i] < 58.0
        
        # === WILLIAMS %R MOMENTUM ===
        wr = wr_smooth[i]
        
        # === VOLUME CONFIRMATION ===
        vol_ok = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position and is_tradeable:
            # LONG: Price breaks above 1d Donchian upper + Williams %R shows momentum
            if donch_breakout_up and price_above_sma:
                if wr < -50:  # Momentum building (not overbought yet)
                    desired_signal = SIZE
            
            # SHORT: Price breaks below 1d Donchian lower
            if donch_breakout_down and not price_above_sma:
                if wr > -50:  # Bearish momentum
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        if in_position:
            atr_entry = atr_14[entry_bar] if entry_bar >= 0 else atr_14[i]
            
            if position_side > 0:
                stop_loss = entry_price - 2.5 * atr_entry
                if low[i] < stop_loss:
                    desired_signal = 0.0
            
            if position_side < 0:
                stop_loss = entry_price + 2.5 * atr_entry
                if high[i] > stop_loss:
                    desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 8 bars = 2 days) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 8:
            # Exit on momentum reversal
            if position_side > 0 and wr > -20:
                desired_signal = 0.0
            if position_side < 0 and wr < -80:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals