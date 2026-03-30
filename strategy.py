#!/usr/bin/env python3
"""
Experiment #024: 4h Williams Alligator + 1d SMA Trend + Donchian Breakout + Volume

HYPOTHESIS: Williams Alligator (EMA 5/8/13) provides smooth trend detection.
Combined with 1d SMA macro trend filter and 4h Donchian(20) breakout confirmation,
this captures the middle of trends without chasing false breakouts.
Volume spike confirms institutional participation.

WHY IT SHOULD WORK:
- 2021 bull: Alligator spread + Donchian breakout in uptrend = strong longs
- 2022 bear: Alligator convergence + Donchian breakdown in downtrend = strong shorts  
- 2025 range: Alligator coiled = no trades (avoid whipsaws)

EXPECTED TRADES: 100-200 total over 4 years (25-50/year)
Size: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_alligator_donchian_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_ema(data, period):
    """Exponential Moving Average"""
    return pd.Series(data).ewm(span=period, min_periods=period, adjust=False).mean().values

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

def calculate_donchian(high, low, period=20):
    """Donchian channel"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    middle = (upper + pd.Series(low).rolling(window=period, min_periods=period).min().values) / 2
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d_50 = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    # === Williams Alligator (EMA 5/8/13) ===
    jaw = calculate_ema(close, 13)    # Blue line
    teeth = calculate_ema(close, 8)  # Red line  
    lips = calculate_ema(close, 5)   # Green line
    
    # === Donchian 20 for breakout confirmation ===
    dc_upper, dc_middle, dc_lower = calculate_donchian(high, low, period=20)
    
    # === Volume ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === ATR for stoploss ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
    
    warmup = 50
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === ALLIGATOR TREND ===
        # Bullish: Lips > Teeth > Jaw (alligator mouth open up)
        alligator_bullish = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        # Bearish: Lips < Teeth < Jaw (alligator mouth open down)
        alligator_bearish = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # === 1D SMA TREND ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_broken_up = close[i] > dc_upper[i] if not np.isnan(dc_upper[i]) else False
        donchian_broken_down = close[i] < dc_lower[i] if not np.isnan(dc_lower[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MINIMUM HOLD: 2 bars (8h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP (2.5x ATR) ===
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
            
            # Opposite trend exit
            if position_side > 0 and (htf_bearish or not alligator_bullish) and min_hold:
                stop_hit = True
            if position_side < 0 and (htf_bullish or not alligator_bearish) and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: All bullish + 1d uptrend + Donchian breakout + volume
            if alligator_bullish and htf_bullish and donchian_broken_up and vol_spike:
                in_position = True
                position_side = 1
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: All bearish + 1d downtrend + Donchian breakdown + volume
            elif alligator_bearish and htf_bearish and donchian_broken_down and vol_spike:
                in_position = True
                position_side = -1
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals