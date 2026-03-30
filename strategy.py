#!/usr/bin/env python3
"""
Experiment #005 v2: 12h Donchian Breakout + Williams %R + ATR Regime

HYPOTHESIS: Donchian breakouts capture institutional momentum when price escapes
consolidation. Williams %R filters for reversals at extremes (oversold/overbought).
ATR regime confirms trending volatility (avoiding ranging chop).

WHY 12h: 2x slower than 4h = ~2x fewer trades = less fee drag.
Donchian(20) on 12h captures ~5-10 day swing extremes.

WHY IT WORKS IN BULL AND BEAR: Breakouts work in both directions.
1d EMA filter ensures we're trading WITH the higher timeframe trend.
Volume spike confirms institutional participation.
Williams %R < -80 for longs = catch reversals near swing lows.
Williams %R > -20 for shorts = catch reversals near swing highs.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_willr_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

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
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_ma20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Williams %R (14) - momentum confirmation
    willr_14 = calculate_williams_r(high, low, close, period=14)
    
    # Donchian channel (20) - breakout structure
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio
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
    
    warmup = 100
    
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
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === WILLIAMS %R - momentum confirmation ===
        willr = willr_14[i]
        willr_oversold = willr < -80  # Strong reversal zone for longs
        willr_overbought = willr > -20  # Strong reversal zone for shorts
        
        # === VOLUME confirmation ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ATR REGIME - trending volatility filter ===
        atr_expanding = atr_14[i] > atr_ma20[i]
        
        # Previous bar's Donchian for breakout (no look-ahead)
        prev_donchian_high = donchian_high[i - 1]
        prev_donchian_low = donchian_low[i - 1]
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above 20-bar high with confluence ===
            # Conditions: 1d trend UP + willr oversold + vol spike + ATR expanding
            if price_above_1d_ema and willr_oversold and vol_spike and atr_expanding:
                # Price breaks above previous Donchian high
                if close[i] > prev_donchian_high:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below 20-bar low with confluence ===
            # Conditions: 1d trend DOWN + willr overbought + vol spike + ATR expanding
            if not price_above_1d_ema and willr_overbought and vol_spike and atr_expanding:
                # Price breaks below previous Donchian low
                if close[i] < prev_donchian_low:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR from entry) ===
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
        
        # === MINIMUM HOLD (2 bars = 1 day to reduce churn) ===
        bars_held = i - entry_bar
        if in_position and bars_held < 2:
            # Force hold for minimum 2 bars
            if position_side > 0:
                desired_signal = SIZE
            else:
                desired_signal = -SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals