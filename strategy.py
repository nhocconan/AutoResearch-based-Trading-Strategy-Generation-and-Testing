#!/usr/bin/env python3
"""
Experiment #024: 12h Camarilla Mean Reversion + 1d SMA Trend + Volume

HYPOTHESIS: Camarilla S3/R3 levels are statistically significant reversal
points. Combined with 1d SMA trend filter and volume confirmation:
- Bull market (2021): Fades R3 touches in uptrend for mean-reversion longs
- Bear market (2022): Fades S3 touches in downtrend for shorts  
- Range market (2025): Bounces between S3/R3 cleanly

KEY INSIGHT: Previous strategies chased breakouts. This fades extremes at
known pivot levels - opposite approach, different edge, better for range.

Expected: 75-150 trades over 4 years (tight entry = fewer but better).
Size: 0.25-0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_1d_sma_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """
    Classic Camarilla pivot levels.
    R3 = close + (high - low) * 1.1
    S3 = close - (high - low) * 1.1
    R4 = close + (high - low) * 1.2
    S4 = close - (high - low) * 1.2
    """
    rng = high - low
    r4 = close + rng * 1.2
    r3 = close + rng * 1.1
    s3 = close - rng * 1.1
    s4 = close - rng * 1.2
    return r4, r3, s3, s4

def calculate_atr(high, low, close, period=14):
    """Average True Range with EWM smoothing"""
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
    """Donchian channel for structural reference"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # === 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    r4, r3, s3, s4 = calculate_camarilla(high, low, close)
    
    # Donchian 20 for structural reference
    dc_upper_20, dc_lower_20 = calculate_donchian(high, low, period=20)
    
    # Volume analysis
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]) or np.isnan(dc_upper_20[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === TREND DETECTION (1d SMA) ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # Price near Donchian extremes (confluence with Camarilla)
        near_dc_high = close[i] > dc_upper_20[i] * 0.99
        near_dc_low = close[i] < dc_lower_20[i] * 1.01
        
        # === CAMARILLA SIGNALS ===
        # Long: price AT or BELOW S3 (oversold) + bullish 1d trend
        at_s3_long = close[i] <= s3[i] * 1.01 and close[i] >= s3[i] * 0.97
        # Short: price AT or ABOVE R3 (overbought) + bearish 1d trend
        at_r3_short = close[i] >= r3[i] * 0.99 and close[i] <= r3[i] * 1.03
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MINIMUM HOLD: 2 bars (24h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        if in_position:
            if position_side > 0:
                stop_hit = low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                stop_hit = high[i] > (lowest_since_entry + 2.5 * entry_atr)
            
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
            continue
        
        # === NEW POSITIONS ===
        # LONG: At S3 + volume spike + 1d bullish
        if at_s3_long and vol_spike and htf_bullish:
            in_position = True
            position_side = 1
            entry_atr = atr_14[i]
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        
        # LONG: At S3 without spike but strong trend + near DC low
        elif at_s3_long and htf_bullish and near_dc_low:
            in_position = True
            position_side = 1
            entry_atr = atr_14[i]
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE * 0.75  # Reduced size without volume spike
        
        # SHORT: At R3 + volume spike + 1d bearish
        elif at_r3_short and vol_spike and htf_bearish:
            in_position = True
            position_side = -1
            entry_atr = atr_14[i]
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        
        # SHORT: At R3 without spike but strong trend + near DC high
        elif at_r3_short and htf_bearish and near_dc_high:
            in_position = True
            position_side = -1
            entry_atr = atr_14[i]
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE * 0.75  # Reduced size without volume spike
        
        else:
            signals[i] = 0.0
    
    return signals