#!/usr/bin/env python3
"""
Experiment #028: 12h Camarilla Pivot Breakout + Volume Spike + 1d Choppiness Regime

HYPOTHESIS: Camarilla pivot levels are mathematically precise support/resistance 
that institutions use. Combined with 1d Choppiness Index to avoid range-bound 
markets and volume spike for institutional confirmation, this captures high-probability
reversals at key levels.

WHY 12h: Target 50-150 total trades over 4 years = 12-37/year. Slow enough to 
reduce fee drag, fast enough for meaningful signals. Uses HTF 1d for regime.

WHY IT WORKS IN BULL AND BEAR: Camarilla pivots work in both directions:
- Long: price touches S3/S4 with vol spike + 1d not choppy
- Short: price touches R3/R4 with vol spike + 1d not choppy
Symmetrical logic for up and down markets.

KEY FIXES from #027:
- Uses CAMARILLA pivots (more precise than Donchian)
- Uses 1d Choppiness (proven pattern from DB: 1.47 Sharpe)
- Tighter entry: requires BOTH vol spike AND non-choppy regime
- Size: 0.25
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_chop_1d_v1"
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
    """Choppiness Index - values below 38.2 = trending, above 61.8 = choppy"""
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

def calculate_camarilla_pivots(high, low, close):
    """
    Camarilla Pivot Levels (8 levels)
    S1 = close - (high - low) * 1.1 / 12
    S2 = close - (high - low) * 1.1 / 6
    S3 = close - (high - low) * 1.1 / 4
    S4 = close - (high - low) * 1.1 / 2
    R1 = close + (high - low) * 1.1 / 12
    R2 = close + (high - low) * 1.1 / 6
    R3 = close + (high - low) * 1.1 / 4
    R4 = close + (high - low) * 1.1 / 2
    """
    n = len(close)
    pivot_range = (high - low) * 1.1 / 12
    
    s1 = close - pivot_range
    s2 = close - (high - low) * 1.1 / 6
    s3 = close - (high - low) * 1.1 / 4
    s4 = close - (high - low) * 1.1 / 2
    
    r1 = close + pivot_range
    r2 = close + (high - low) * 1.1 / 6
    r3 = close + (high - low) * 1.1 / 4
    r4 = close + (high - low) * 1.1 / 2
    
    return {
        'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4,
        'R1': r1, 'R2': r2, 'R3': r3, 'R4': r4
    }

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Choppiness for regime detection
    chop_1d_raw = calculate_choppiness(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # 1d SMA50 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    pivots = calculate_camarilla_pivots(high, low, close)
    
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
    
    warmup = 100  # Need enough for indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME CHECK (1d Choppiness) ===
        # CHOP < 61.8 = not too choppy (can trade)
        # CHOP < 50 = trending (prefer momentum trades)
        # CHOP > 61.8 = very choppy (skip)
        chop_regime_ok = chop_1d_aligned[i] < 61.8
        is_trending = chop_1d_aligned[i] < 50.0
        
        if not chop_regime_ok and not in_position:
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === CAMARILLA LEVEL TOUCH ===
        s3 = pivots['S3'][i]
        s4 = pivots['S4'][i]
        r3 = pivots['R3'][i]
        r4 = pivots['R4'][i]
        
        # Detect touch of outer levels (S4 or R4) - more extreme = higher conviction
        touched_s4 = low[i] <= s4
        touched_r4 = high[i] >= r4
        touched_s3 = low[i] <= s3
        touched_r3 = high[i] >= r3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Price touches S4 with volume + non-choppy regime ===
            # S4 is extreme support - bounce expected
            if touched_s4 and vol_spike:
                desired_signal = SIZE
            
            # === LONG ENTRY: Price touches S3 with volume + trending regime ===
            if touched_s3 and vol_spike and is_trending:
                desired_signal = SIZE * 0.5  # Smaller size for less extreme
            
            # === SHORT ENTRY: Price touches R4 with volume + non-choppy regime ===
            if touched_r4 and vol_spike:
                desired_signal = -SIZE
            
            # === SHORT ENTRY: Price touches R3 with volume + trending regime ===
            if touched_r3 and vol_spike and is_trending:
                desired_signal = -SIZE * 0.5  # Smaller size for less extreme
        
        # === STOPLOSS CHECK (2.0 ATR) ===
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
        
        # === TIME-BASED EXIT (hold at least 4 bars = 2 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 4:
            # Exit if price reaches opposite Camarilla level
            if position_side > 0:
                # Take profit at R1 or R2
                if close[i] >= r1:
                    desired_signal = 0.0
            if position_side < 0:
                # Take profit at S1 or S2
                if close[i] <= s1:
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