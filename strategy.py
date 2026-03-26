Looking at the experiment history, I see:
1. Overtrading failures (524, 1321 trades) - fee drag destroys returns
2. Too few trades (0-8 trades) - statistically unreliable
3. Negative Sharpe despite having trades - poor entry quality

The DB shows that **12h/4h HMA + Donchian + Volume** works (test Sharpe 1.38). Let me try a different angle: **Camarilla Pivots** combined with **Williams Alligator** (not yet tried) for trend direction.

**Hypothesis**: Williams Alligator (jaw/teeth/lips on median price) identifies institutional trend direction. When Alligator is "opening" (bullish alignment) AND price touches a Camarilla S3/R3 level with volume spike, enter. This combination captures reversals at key pivot levels while avoiding whipsaws.
#!/usr/bin/env python3
"""
Experiment #021: Williams Alligator + Camarilla Pivots + Volume Spike

HYPOTHESIS: Williams Alligator identifies institutional trend direction via
three aligned SMAs on median price (jaw/teeth/lips). Camarilla pivots mark
key support/resistance where reversals occur. When Alligator shows alignment
AND price touches Camarilla S3/R3 with volume spike, high-probability entry.
Works in both bull (buy S3 bounces) and bear (sell R3 rejections).

TIMEFRAME: 4h primary
HTF: 12h Alligator for trend direction
TARGET: 75-150 total trades over 4 years (18-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_alligator_camarilla_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_alligator(high, low, jaw_period=13, teeth_period=8, lips_period=5):
    """Williams Alligator - three SMAs on median price (H+L)/2"""
    median = (high + low) / 2.0
    
    jaw = pd.Series(median).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(median).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(median).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    return jaw, teeth, lips

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

def calculate_camarilla(high, low, close):
    """Camarilla pivot levels"""
    n = len(close)
    pivot = (high + low + close) / 3.0
    
    r4 = close + (high - low) * 1.1 / 2.0
    r3 = close + (high - low) * 1.1 / 4.0
    r2 = close + (high - low) * 1.1 / 6.0
    r1 = close + (high - low) * 1.1 / 12.0
    s1 = close - (high - low) * 1.1 / 12.0
    s2 = close - (high - low) * 1.1 / 6.0
    s3 = close - (high - low) * 1.1 / 4.0
    s4 = close - (high - low) * 1.1 / 2.0
    
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 12h HTF data ONCE ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h Alligator for trend direction
    jaw_12h, teeth_12h, lips_12h = calculate_alligator(
        df_12h['high'].values, df_12h['low'].values,
        jaw_period=13, teeth_period=8, lips_period=5
    )
    
    # Align 12h to 4h
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Calculate local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    jaw_4h, teeth_4h, lips_4h = calculate_alligator(high, low)
    r4, r3, r2, r1, pivot, s1, s2, s3, s4 = calculate_camarilla(high, low, close)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Check 4h Alligator ready
        if np.isnan(jaw_4h[i]) or np.isnan(teeth_4h[i]) or np.isnan(lips_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Check 12h Alligator ready
        if np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === ALLIGATOR ALIGNMENT ===
        # 4h Alligator: jaw > teeth > lips = bullish (alligator eating)
        bullish_4h = (jaw_4h[i] > teeth_4h[i]) and (teeth_4h[i] > lips_4h[i])
        # 4h Alligator: jaw < teeth < lips = bearish (alligator eating)
        bearish_4h = (jaw_4h[i] < teeth_4h[i]) and (teeth_4h[i] < lips_4h[i])
        
        # 12h Alligator confirmation
        bullish_12h = (jaw_12h_aligned[i] > teeth_12h_aligned[i]) and (teeth_12h_aligned[i] > lips_12h_aligned[i])
        bearish_12h = (jaw_12h_aligned[i] < teeth_12h_aligned[i]) and (teeth_12h_aligned[i] < lips_12h_aligned[i])
        
        # === CAMARILLA TOUCH ===
        # Price touches S3 (for longs) or R3 (for shorts)
        touch_s3 = (low[i] <= s3[i]) and (close[i] >= s3[i] - 0.5 * atr_14[i])
        touch_r3 = (high[i] >= r3[i]) and (close[i] <= r3[i] + 0.5 * atr_14[i])
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === DONCHIAN CHANNEL for structure ===
        # Simple 20-period for context
        upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
        mid_20 = (upper_20 + lower_20) / 2.0
        
        # Price near middle or above for longs, near middle or below for shorts
        near_mid_or_above = close[i] >= mid_20[i] * 0.98
        near_mid_or_below = close[i] <= mid_20[i] * 1.02
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Alligator bullish + price touches S3 + volume spike
            if bullish_4h and bullish_12h:
                if touch_s3 and vol_spike:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Alligator bearish + price touches R3 + volume spike
            if bearish_4h and bearish_12h:
                if touch_r3 and vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT: Opposite Alligator signal ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Exit long if Alligator flips bearish
            if bearish_4h:
                exit_triggered = True
            # Or price hits R3
            if high[i] >= r3[i]:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Exit short if Alligator flips bullish
            if bullish_4h:
                exit_triggered = True
            # Or price hits S3
            if low[i] <= s3[i]:
                exit_triggered = True
        
        if exit_triggered:
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
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals