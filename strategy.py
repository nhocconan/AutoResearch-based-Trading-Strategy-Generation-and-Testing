#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian Breakout + Weekly EMA + Choppiness Regime

HYPOTHESIS: Price channel breakouts (Donchian) are the most reliable structural
signals. Using 1w EMA for trend direction filters out false breakouts in choppy
markets. Choppiness Index > 38.2 means trending (good for breakouts), < 38.2 means
choppy (avoid signals).

WHY 12h: Slow enough to capture multi-day swings with 30-50 trades/year.
Weekly EMA aligns with institutional trend direction.

WHY IT WORKS BOTH MARKETS:
- Bull: Buy breakouts above weekly EMA (captures rallies)
- Bear: Short breakouts below weekly EMA (captures breakdowns)
- Range: Choppiness filter keeps us flat during chop

TARGET: 75-200 total trades over 4 years. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_ema1w_chop_v1"
timeframe = "12h"
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
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = very choppy (range-bound, mean reversion)
    CHOP < 38.2 = trending (good for trend following)
    Values between are neutral
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        sum_tr = 0.0
        for j in range(i - period + 1, i + 1):
            sum_tr += max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
        
        if highest_high - lowest_low > 1e-10:
            chop[i] = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA for trend direction
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels (20 bars = 10 days on 12h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
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
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1w EMA21) ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        
        # === REGIME FILTER (Choppiness) ===
        # CHOP < 38.2 = trending, good for breakouts
        is_trending = chop[i] > 38.2
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian breakout from previous bar (no look-ahead)
        prev_donchian_high = donchian_high[i - 1]
        prev_donchian_low = donchian_low[i - 1]
        
        # Breakout detection: price closes above/below previous Donchian
        bull_breakout = close[i] > prev_donchian_high
        bear_breakout = close[i] < prev_donchian_low
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Bull breakout + above weekly EMA + trending + volume ===
            if bull_breakout and price_above_1w_ema and is_trending and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Bear breakout + below weekly EMA + trending + volume ===
            if bear_breakout and not price_above_1w_ema and is_trending and vol_spike:
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
        
        # === MINIMUM HOLD (3 bars = 1.5 days to avoid chop) ===
        bars_held = i - entry_bar
        
        # === TAKE PROFIT (1.5R) ===
        if in_position and bars_held >= 3:
            if position_side > 0:
                profit_target = entry_price + 1.5 * entry_atr
                if high[i] >= profit_target:
                    desired_signal = SIZE / 2  # Half position
            elif position_side < 0:
                profit_target = entry_price - 1.5 * entry_atr
                if low[i] <= profit_target:
                    desired_signal = -SIZE / 2  # Half position
        
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
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals