#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian + Williams Alligator + 1d EMA (Tighter Filters)

HYPOTHESIS: Previous Camarilla strategy (275 trades) overtraded due to multiple
Camarilla levels triggering. Replace with:
- Donchian(24) for structure (less noise than Camarilla multi-levels)
- Williams Alligator for momentum confirmation (more selective than volume alone)
- 1d EMA for trend alignment
- Choppiness filter to skip range-bound periods

WHY IT WORKS: Donchian breakout captures institutional moves, Alligator alignment
ensures momentum is confirmed, 1d filter avoids countertrend trades.

TARGET: 100-150 total trades over 4 years. Previous had 275 - this version
tightens with Alligator alignment requirement (both lips>teeth AND teeth>jaw).

12h timeframe = ~2922 bars/year, target ~25-35 trades/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_alligator_ema50_1d_v1"
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
    """Choppiness Index - lower = trending, higher = ranging"""
    n = len(close)
    chop = np.full(n, 61.8)
    
    for i in range(period, n):
        sum_tr = 0.0
        for j in range(period):
            idx = i - j
            tr = max(high[idx] - low[idx], abs(high[idx] - close[idx-1]), abs(low[idx] - close[idx-1]))
            sum_tr += tr
        
        highest_high = max(high[i-period+1:i+1])
        lowest_low = min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10:
            chop[i] = 100 * np.log(sum_tr / range_hl) / np.log(period)
    
    return chop


def calculate_alligator(high, low, close):
    """Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3)"""
    n = len(close)
    median = (high + low) / 2.0
    
    # Calculate SMAs
    sma13 = pd.Series(median).rolling(window=13, min_periods=13).mean().values
    sma8 = pd.Series(median).rolling(window=8, min_periods=8).mean().values
    sma5 = pd.Series(median).rolling(window=5, min_periods=5).mean().values
    
    # Apply Alligator offsets (shift forward = values appear "later")
    jaw = np.full(n, np.nan)   # SMA13 shifted 8 bars forward
    teeth = np.full(n, np.nan)  # SMA8 shifted 5 bars forward
    lips = np.full(n, np.nan)   # SMA5 shifted 3 bars forward
    
    for i in range(8, n):
        jaw[i] = sma13[i - 8] if (i - 8) >= 0 else np.nan
    for i in range(5, n):
        teeth[i] = sma8[i - 5] if (i - 5) >= 0 else np.nan
    for i in range(3, n):
        lips[i] = sma5[i - 3] if (i - 3) >= 0 else np.nan
    
    return jaw, teeth, lips


def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d_raw = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_raw)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # Volume ratio (20-bar SMA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Donchian channels (24-bar = 12 days)
    donchian_period = 24
    donchian_upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_middle = (donchian_upper + pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values) / 2.0
    donchian_lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    entry_bar = 0
    
    # Warmup: 150 bars to cover Alligator offset (21) + EMA alignment buffer
    warmup = 150
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Choppiness filter (skip if choppy/ranging)
        if chop[i] > 61.8:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === ALLIGATOR ALIGNMENT ===
        # Long: lips > teeth > jaw (bullish stack)
        # Short: lips < teeth < jaw (bearish stack)
        alligator_bull = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        alligator_bear = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Alligator "mouth open" = lines separated (not tangled)
        alligator_spread = abs(jaw[i] - lips[i]) if not np.isnan(jaw[i]) and not np.isnan(lips[i]) else 0
        jaw_range = jaw[i] * 0.01 if not np.isnan(jaw[i]) else 1
        alligator_active = alligator_spread > jaw_range
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.8
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_upper = donchian_upper[i - 1] if i > 0 else high[i]
        prev_donchian_lower = donchian_lower[i - 1] if i > 0 else low[i]
        
        # Price breaks above previous Donchian high
        donchian_breakout_up = close[i] > prev_donchian_upper
        # Price breaks below previous Donchian low
        donchian_breakout_down = close[i] < prev_donchian_lower
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Donchian breakout + Alligator bullish + volume + 1d trend ===
            if (donchian_breakout_up and alligator_bull and alligator_active and 
                vol_spike and price_above_1d_ema):
                desired_signal = SIZE
            
            # === SHORT: Donchian breakdown + Alligator bearish + volume + 1d trend ===
            if (donchian_breakout_down and alligator_bear and alligator_active and 
                vol_spike and not price_above_1d_ema):
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR) ===
        if in_position:
            if position_side > 0:
                if low[i] < stop_price:
                    desired_signal = 0.0
            else:
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD (6 bars = 3 days to reduce churn) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 6:
            # Exit if Alligator flips against position
            if position_side > 0 and alligator_bear:
                desired_signal = 0.0
            if position_side < 0 and alligator_bull:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals