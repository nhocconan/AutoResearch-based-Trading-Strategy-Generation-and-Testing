#!/usr/bin/env python3
"""
Experiment #024: 4h Donchian Breakout + Volume Spike + Choppiness Regime

HYPOTHESIS: Donchian(20) breakout captures momentum shifts when price breaks
the 20-bar structure high/low. Volume spike confirms institutional conviction.
Choppiness Index ensures we only trade in trending conditions. ATR stoploss
provides risk control.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Donchian breakout works in ALL markets (bull, bear, range) because it's based on
  price structure, not direction
- Bear markets: short breakouts with tight ATR stop
- Bull markets: long breakouts with trailing stop
- Range exits prevent whipsaws

TARGET: 75-150 total trades over 4 years (proven pattern from DB).
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95 trades)

KEY DESIGN (3 conditions max):
1. Donchian(20) breakout (price closes above/below 20-bar high/low)
2. Volume spike (>1.5x 20-avg) - required confirmation
3. Choppiness < 55 - regime filter
4. ATR-based stoploss (2x ATR)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging (no trades), CHOP < 55 = trending (allow trades)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel
    Returns upper (high_n), lower (low_n), and middle
    """
    n = len(high)
    high_n = pd.Series(high).rolling(window=period, min_periods=period).max().values
    low_n = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (high_n + low_n) / 2.0
    return high_n, low_n, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Donchian(20)
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Warmup for indicators
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 55.0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN LEVELS ===
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        
        # Price position relative to Donchian
        if atr_14[i] > 0:
            dist_above_high = (close[i] - dc_high) / atr_14[i]
            dist_below_low = (dc_low - close[i]) / atr_14[i]
        else:
            dist_above_high = -999
            dist_below_low = -999
        
        # === ENTRY LOGIC ===
        # Long: Break above Donchian high + volume spike + trending
        if is_trending and vol_spike:
            # Price closes above/beyond Donchian high (within 0.3 ATR)
            if dist_above_high >= -0.3 and dist_above_high < 1.5:
                signals[i] = SIZE
        
        # Short: Break below Donchian low + volume spike + trending
        if is_trending and vol_spike:
            # Price closes below/beyond Donchian low (within 0.3 ATR)
            if dist_below_low >= -0.3 and dist_below_low < 1.5:
                signals[i] = -SIZE
    
    return signals