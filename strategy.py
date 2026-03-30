#!/usr/bin/env python3
"""
Experiment #006: 4h ATR Volatility Regime Breakout + Volume + 1d EMA Trend

HYPOTHESIS: Markets alternate between trending (high ATR) and ranging (low ATR) regimes.
By ONLY entering during trending regimes (ATR ratio > 1.5x) AND when price breaks
a tight Donchian channel AND volume confirms, we catch the start of big moves.

WHY IT WORKS IN BULL AND BEAR: In bull markets, trending ATR regime + EMA up = long
breakouts. In bear markets, trending ATR regime + EMA down = short breakdowns.
The ATR regime filter prevents false breakouts during choppy periods (2022 crash).

KEY INSIGHT FROM DB: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 achieved
test Sharpe 1.47 with 95 trades. The key was TIGHT entries with confluence.

TARGET: 50-100 total trades over 4 years = 12-25/year. HARD MAX: 150.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_atr_volregime_donchian_1d_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper, middle, lower arrays"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    middle = (upper + pd.Series(low).rolling(window=period, min_periods=period).min().values) / 2
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA100 for trend (slower = more stable trend filter)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=100, min_periods=100, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # ATR regime: current ATR vs 30-bar average (1.5x = trending)
    atr_ratio = atr_14 / np.where(atr_30 > 0, atr_30, 1)
    
    # Donchian 20 (same as DB winner)
    dc_upper, dc_mid, dc_lower = calculate_donchian(high, low, period=20)
    
    # Volume ratio (20-bar MA)
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 150  # Need enough for EMA100 alignment buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(atr_ratio[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA100) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === ATR REGIME: Only trade in trending markets ===
        is_trending = atr_ratio[i] > 1.5
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.8
        
        # Donchian levels from previous bar (no look-ahead)
        dc_upper_prev = dc_upper[i - 1]
        dc_lower_prev = dc_lower[i - 1]
        dc_mid_prev = dc_mid[i - 1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === TIGHT ENTRY: All conditions must align ===
            # 1. Trending ATR regime
            # 2. Volume spike confirmation
            # 3. Price above 1d EMA (bull) or below (bear)
            # 4. Breakout through Donchian channel
            
            # LONG: Trending + Volume + EMA up + Breakout above DC upper
            if is_trending and vol_spike and price_above_1d_ema:
                if close[i] > dc_upper_prev and low[i] > dc_upper_prev:
                    # Breakout confirmed, enter long
                    desired_signal = SIZE
            
            # SHORT: Trending + Volume + EMA down + Breakdown below DC lower
            if is_trending and vol_spike and not price_above_1d_ema:
                if close[i] < dc_lower_prev and high[i] < dc_lower_prev:
                    # Breakdown confirmed, enter short
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing - tighter for 4h) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars (16h) to avoid fee churn ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 4:
            # Take profit at Donchian mid or opposite band
            if position_side > 0 and close[i] >= dc_mid_prev:
                desired_signal = 0.0
            if position_side < 0 and close[i] <= dc_mid_prev:
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
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals