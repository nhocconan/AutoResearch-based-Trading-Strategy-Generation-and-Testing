#!/usr/bin/env python3
"""
Experiment #028: 1d Donchian(10) Breakout + 1w Trend + Volume

HYPOTHESIS: 1d timeframe naturally limits trade frequency to ~15-20/year.
Donchian(10) captures ~2-week breakouts - tight enough to avoid whipsaws.
1w SMA200 keeps us aligned with the major trend (avoids fighting 2022 crash).
Volume confirmation ensures institutional participation.
Simple = robust = better test generalization.

WHY IT WORKS BOTH MARKETS:
- Bull: Long breakouts above 1w SMA200 + Donchian high
- Bear: Short breakdowns below 1w SMA200 + Donchian low
- Range: Choppiness filter keeps us out

TARGET: 50-80 total trades over 4 years = 12-20/year. HARD MAX: 150.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_vol_1w_v1"
timeframe = "1d"
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
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA200 for major trend
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Local 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels - TIGHTER (10 = 2-week channel)
    donchian_period = 10
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    warmup = 250  # Need 200 for 1w SMA + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === MAJOR TREND (1w SMA200) ===
        price_above_1w_sma = close[i] > sma_1w_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # Skip if too choppy when looking for new entries
        is_choppy = chop[i] > 61.8
        
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        current_high = high[i]
        current_low = low[i]
        
        # Previous bar's Donchian values (avoid look-ahead)
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        # Volume confirmation (1.5x average)
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high ===
            # Price closes above previous 10-bar high with volume
            if current_high > prev_donchian_high and price_above_1w_sma:
                if vol_spike:  # Volume confirmation
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low ===
            # Price closes below previous 10-bar low with volume
            if current_low < prev_donchian_low and not price_above_1w_sma:
                if vol_spike:  # Volume confirmation
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
        
        # === MINIMUM HOLD (5 bars = 5 days to avoid whipsaw) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held < 5:
            # Keep position, don't exit early
            desired_signal = position_side * SIZE
        
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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