#!/usr/bin/env python3
"""
Experiment #022: 4h Donchian(20) Breakout + TRIX Momentum + Volume Filter

HYPOTHESIS: Donchian(20) breakout captures momentum at key structural levels.
Adding TRIX(12) momentum filter ensures entries align with directional momentum.
Volume confirmation filters noise breakouts. HTF EMA21 trend alignment ensures
entries work in BOTH bull and bear markets.

WHY IT WORKS: Donchian channels are widely watched institutional levels.
A breakout above/below these levels with momentum confirmation catches trends
while TRIX filters out whipsaws in ranging markets.

TARGET: 100-200 total trades over 4 years = 25-50/year.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_trix_vol_ema21_1d_v1"
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

def calculate_trix(prices, period=12):
    """TRIX - Triple EMA Oscillator"""
    close = prices.values if hasattr(prices, 'values') else prices
    
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # TRIX is the rate of change of the triple EMA
    trix = ema3.pct_change(period) * 100
    return trix.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA21 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 bars)
    roll_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    roll_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = roll_high.values
    donchian_low = roll_low.values
    
    # TRIX momentum
    close_series = pd.Series(close)
    trix = calculate_trix(close_series, period=12)
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR regime (ATR percentile)
    atr_ma = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_14 / np.where(atr_ma > 0, atr_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 60  # Need enough for indicators
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Donchian not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if TRIX not ready
        if np.isnan(trix[i]) or np.isnan(trix[i-1] if i > 0 else 0):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA21) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === TRIX MOMENTUM (positive = uptrend momentum) ===
        trix_positive = trix[i] > 0
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        # === ATR REGIME (ATR ratio > 1.2 = trending/high vol, avoid low vol) ===
        atr_trending = atr_ratio[i] > 0.8  # Low ATR = ranging, skip
        
        # === DONCHIAN BREAKOUT ===
        donchian_broken_up = close[i] > donchian_high[i]
        donchian_broken_down = close[i] < donchian_low[i]
        
        # Previous bar's Donchian (no look-ahead)
        prev_donchian_high = donchian_high[i-1]
        prev_donchian_low = donchian_low[i-1]
        prev_donchian_broken_up = close[i-1] > prev_donchian_high
        prev_donchian_broken_down = close[i-1] < prev_donchian_low
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Donchian breakout + HTF trend up + momentum confirmation ===
            if price_above_1d_ema and trix_positive and atr_trending:
                # Breakout on THIS bar or PREVIOUS bar (for late entries)
                if donchian_broken_up or prev_donchian_broken_up:
                    if vol_spike:  # Volume confirmation
                        desired_signal = SIZE
            
            # === SHORT: Donchian breakdown + HTF trend down + momentum confirmation ===
            if not price_above_1d_ema and not trix_positive and atr_trending:
                if donchian_broken_down or prev_donchian_broken_down:
                    if vol_spike:
                        desired_signal = -SIZE
        
        # === HOLD PERIOD (minimum 2 bars to avoid churn) ===
        bars_held = i - entry_bar
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === EXIT ON MOMENTUM REVERSAL (after min hold) ===
        if in_position and bars_held >= 2:
            # Exit long if TRIX flips negative
            if position_side > 0 and trix_cross_down:
                desired_signal = 0.0
            # Exit short if TRIX flips positive
            if position_side < 0 and trix_cross_up:
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