#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + ATR Volatility Regime + 1d EMA Trend

HYPOTHESIS: Donchian(20) breakouts work best when volatility is COMPRESSED
(ATR ratio < 1.5 = choppy market about to break out). Exit when volatility
EXPANDS (ATR ratio > 2.0 = trend exhaustion).

WHY IT WORKS IN BULL AND BEAR:
- Long breakouts above 1d EMA in bull markets
- Short breakouts below 1d EMA in bear markets
- ATR filter adapts to both regimes

TARGET: 100-200 total trades over 4 years = 25-50/year. HARD MAX: 300.
Signal size: 0.30.

Previous failures learned:
- 12h timeframe → fails (use 4h)
- CHOP filter → too restrictive (use ATR ratio)
- Overtrading (426+ trades) → add min hold + vol spike requirement
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_atrregime_vol_1d_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # ATR ratio: recent vol vs baseline (filters choppy markets)
    # Safe division: replace 0 with 1 to avoid NaN
    atr_ratio = atr_14 / np.where(atr_30 > 0, atr_30, 1.0)
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    
    # Volume spike confirmation
    vol_spike = vol_ratio > 1.5
    
    # Donchian Channel (20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = 100  # Need enough for all rolling calculations
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if ATR ratio not valid
        if np.isnan(atr_ratio[i]) or atr_ratio[i] > 10.0:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Donchian not ready
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
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
        
        desired_signal = 0.0
        
        # === ATR VOLATILITY REGIME FILTER ===
        # Only enter when volatility is compressed (choppy → breakout)
        low_vol = atr_ratio[i] < 1.5
        
        # === NEW ENTRY LOGIC ===
        if not in_position:
            # Long: close > Donchian upper (breakout) + vol spike + above 1d EMA + low vol
            if low_vol and close[i] > donchian_upper[i] and vol_spike[i] and close[i] > ema_1d_aligned[i]:
                desired_signal = SIZE
            
            # Short: close < Donchian lower + vol spike + below 1d EMA + low vol
            if low_vol and close[i] < donchian_lower[i] and vol_spike[i] and close[i] < ema_1d_aligned[i]:
                desired_signal = -SIZE
        
        # === EXISTING POSITION MANAGEMENT ===
        else:
            # LONG management
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                
                # Stop loss
                if low[i] < trailing_stop:
                    desired_signal = 0.0
                else:
                    # Take profit: close reverts to middle band
                    donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2.0
                    bars_held = i - entry_bar
                    
                    # Min hold = 6 bars (24h) to reduce whipsaw
                    if bars_held >= 6 and close[i] > donchian_mid:
                        desired_signal = 0.0
                    # Early exit if vol expands (trend exhaustion)
                    elif atr_ratio[i] > 2.0:
                        desired_signal = 0.0
            
            # SHORT management
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                
                if high[i] > trailing_stop:
                    desired_signal = 0.0
                else:
                    donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2.0
                    bars_held = i - entry_bar
                    
                    if bars_held >= 6 and close[i] < donchian_mid:
                        desired_signal = 0.0
                    elif atr_ratio[i] > 2.0:
                        desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
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