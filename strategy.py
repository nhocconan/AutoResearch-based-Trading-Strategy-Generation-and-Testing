#!/usr/bin/env python3
"""
Experiment #024: 6h Williams %R Extreme Reversal + 1d SMA + Volume

HYPOTHESIS: Williams %R reaching extreme oversold (<-80) followed by a bounce
indicates institutional buying. The -80/-20 levels are statistically significant
reversal points. Combined with 1d SMA trend filter and volume confirmation,
this catches reversals at key levels without overtrading.

KEY INSIGHT: Previous strategies failed by requiring too many conditions to align.
Williams %R extremes naturally occur ~15-20% of the time, giving us the right
trade frequency. Simple conditions = reliable execution.

TRADE COUNT: 75-150 total over 4 years (18-37/year).
Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_1d_sma_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum indicator for overbought/oversold"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    with np.errstate(divide='ignore', invalid='ignore'):
        williams_r = np.where(
            (highest_high - lowest_low) > 1e-10,
            ((highest_high - close) / (highest_high - lowest_low)) * -100,
            -50  # neutral when no range
        )
    
    return williams_r

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
    """Donchian channel for local trend"""
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
    
    # === 6h indicators ===
    williams_r = calculate_williams_r(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20 for local trend direction
    dc_upper_20, dc_lower_20 = calculate_donchian(high, low, period=20)
    
    # Volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 60
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(williams_r[i]) or np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === TREND DETECTION ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # Williams %R extremes
        is_oversold = williams_r[i] < -80  # Strong oversold
        is_overbought = williams_r[i] > -20  # Strong overbought
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ATR TRAILING STOP ===
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
            
            # Minimum hold: 3 bars (18h)
            min_hold = (i - entry_bar) >= 3
            
            # Opposite trend signal exits
            if position_side > 0 and htf_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and htf_bullish and min_hold:
                stop_hit = True
            
            # Williams %R mean reversion exit
            if position_side > 0 and williams_r[i] > -20 and min_hold:
                stop_hit = True  # Overbought = take profit
            if position_side < 0 and williams_r[i] < -80 and min_hold:
                stop_hit = True  # Oversold = take profit
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Williams %R oversold + bounce starting + 1d uptrend + volume
            # Bounce = current close > previous close (not continuing down)
            bounce = close[i] > close[i-1] if i > 0 else False
            
            if is_oversold and bounce and htf_bullish and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Williams %R overbought + drop starting + 1d downtrend + volume
            drop = close[i] < close[i-1] if i > 0 else False
            
            if is_overbought and drop and htf_bearish and vol_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals