#!/usr/bin/env python3
"""
Experiment #007: 6h Elder Ray + Donchian(10) + Volume Spike

HYPOTHESIS: Elder Ray measures bull/bear power relative to EMA(13).
When bull power turns positive on a breakout, it confirms bullish momentum.
Combined with tight Donchian(10) for quick 6h structure and volume spike
for institutional confirmation, filtered by 1d EMA trend direction.

WHY BOTH MARKETS:
- 2021 bull: Bull power + breakout + volume = strong longs
- 2022 bear: Bear power + breakdown + volume = protective shorts
- 2025 range: 1d EMA filter keeps us flat in chop

TRADE COUNT: 80-160 total over 4 years (target 20-40/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_donchian_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Elder Ray: EMA(13) with bull/bear power
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Smooth bull/bear power
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, min_periods=5, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, min_periods=5, adjust=False).mean().values
    
    # Donchian Channel(10) - 2.5 days on 6h
    donchian_upper = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_lower = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50  # buffer for indicators
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === HTF TREND (1d EMA aligned) ===
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT (prior bar's channel) ===
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = False
        bearish_breakout = False
        
        if not np.isnan(prev_upper) and not np.isnan(prev_lower):
            bullish_breakout = close[i] > prev_upper
            bearish_breakout = close[i] < prev_lower
        
        # === ELDER RAY POWER SHIFT ===
        # Bull power turns positive: bullish momentum emerging
        bull_power_positive = bull_power_smooth[i] > 0
        prev_bull_power = bull_power_smooth[i-1] if i > 0 else 0
        bull_power_cross_up = prev_bull_power <= 0 and bull_power_positive
        
        # Bear power turns negative: bearish momentum emerging
        bear_power_negative = bear_power_smooth[i] < 0
        prev_bear_power = bear_power_smooth[i-1] if i > 0 else 0
        bear_power_cross_down = prev_bear_power >= 0 and bear_power_negative
        
        # === VOLUME SPIKE (>1.5x average) ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MINIMUM HOLD: 3 bars (18h) ===
        min_hold_bars = 3
        min_hold = (i - entry_bar) >= min_hold_bars
        
        # === ATR TRAILING STOP (2.5x ATR from entry high/low) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Exit on trend reversal
            if position_side > 0 and htf_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and htf_bullish and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Bull power turns positive + breakout + volume spike + HTF bullish
            if bull_power_cross_up and bullish_breakout and vol_spike and htf_bullish:
                in_position = True
                position_side = 1
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Bear power turns negative + breakdown + volume spike + HTF bearish
            elif bear_power_cross_down and bearish_breakout and vol_spike and htf_bearish:
                in_position = True
                position_side = -1
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals