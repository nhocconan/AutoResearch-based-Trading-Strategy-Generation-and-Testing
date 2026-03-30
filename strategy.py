#!/usr/bin/env python3
"""
Experiment #006: 12h Donchian(20) Breakout + Volume + 1d EMA Trend + ATR Regime Filter

HYPOTHESIS: Classic Donchian breakout captures major trend shifts.
12h timeframe reduces noise vs 4h (2x fewer trades). Volume confirms breakouts.
1d EMA50 ensures we only trade WITH the major trend.
ATR regime filter eliminates choppy periods (1.2x threshold proven in DB).

WHY IT WORKS IN BOTH BULL AND BEAR:
- Long breakouts above 1d EMA in bull markets (follow the trend)
- Short breakouts below 1d EMA in bear markets (fade rallies)
- Symmetric: same logic, opposite direction, adaptive to regime

WHY IT SHOULD WORK (DB evidence):
- Donchian(20) breakout + volume + ATR stop → SOLUSDT test Sharpe 1.10-1.38
- ATR regime filter was key differentiator in mtf_4h_atrregime_vol_ema50_1d_v1

TARGET: 50-100 total trades over 4 years = 12-25/year. HARD MAX: 150.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_ema50_1d_atrregime_v1"
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
    
    # ATR regime: ATR(7)/ATR(30) > 1.2 indicates trending (not choppy)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_ratio = atr_7 / np.where(atr_14 > 0, atr_14, 1)
    
    # Donchian channels (20 bars = ~10 days on 12h)
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
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Buffer for all indicators
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        # Skip if Donchian not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        is_bullish = close[i] > ema_1d_aligned[i]
        
        # === ATR REGIME FILTER (must be trending, not choppy) ===
        is_trending = atr_ratio[i] > 1.2
        
        # Volume confirmation (1.3x = moderate, not too strict)
        vol_spike = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high + bullish + trending + volume ===
            if is_bullish and is_trending and vol_spike:
                if close[i] > donchian_high[i]:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low + bearish + trending + volume ===
            if not is_bullish and is_trending and vol_spike:
                if close[i] < donchian_low[i]:
                    desired_signal = -SIZE
        
        # === TRAILING STOP (2.5 ATR) ===
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
        
        # === TAKE PROFIT at opposite Donchian band ===
        if in_position and position_side > 0:
            # Take profit if price reaches Donchian low (opposite band)
            if close[i] <= donchian_low[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Take profit if price reaches Donchian high (opposite band)
            if close[i] >= donchian_high[i]:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
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