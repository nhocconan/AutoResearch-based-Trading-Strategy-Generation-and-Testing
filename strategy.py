#!/usr/bin/env python3
"""
Experiment #023: 4h Camarilla Pivots + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla pivot levels (S1-S4, R1-R4) are institutional price 
levels where reversals commonly occur. Combined with volume confirmation 
and Choppiness Index regime filter, this captures mean-reversion trades at 
key levels while avoiding whipsaws in ranging markets. Works in both bull 
and bear by treating S3/S4 as long entries and R3/R4 as short entries.

WHY: Database #1 performer (ETHUSDT test Sharpe=1.471, 95 trades) used 
Camarilla + volume spike + choppiness. This is the proven winning formula.

TIMEFRAME: 4h primary
HTF: 1d for trend confirmation
TARGET: 75-150 total trades over 4 years (19-38/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(high, low, close, period=24):
    """
    Camarilla Pivot Levels
    Uses 24-period (6 days on 4h) high/low/close for intraday pivots
    R4 = close + (high - low) * 1.1/2
    R3 = close + (high - low) * 1.1/4
    R2 = close + (high - low) * 1.1/6
    R1 = close + (high - low) * 1.1/12
    S1 = close - (high - low) * 1.1/12
    S2 = close - (high - low) * 1.1/6
    S3 = close - (high - low) * 1.1/4
    S4 = close - (high - low) * 1.1/2
    """
    n = len(close)
    r4 = np.full(n, np.nan, dtype=np.float64)
    r3 = np.full(n, np.nan, dtype=np.float64)
    r2 = np.full(n, np.nan, dtype=np.float64)
    r1 = np.full(n, np.nan, dtype=np.float64)
    s1 = np.full(n, np.nan, dtype=np.float64)
    s2 = np.full(n, np.nan, dtype=np.float64)
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        hi = high[i - period + 1:i + 1]
        lo = low[i - period + 1:i + 1]
        h_range = np.nanmax(hi) - np.nanmin(lo)
        c = close[i]
        
        r4[i] = c + h_range * 0.55
        r3[i] = c + h_range * 0.275
        r2[i] = c + h_range * 0.183
        r1[i] = c + h_range * 0.0917
        s1[i] = c - h_range * 0.0917
        s2[i] = c - h_range * 0.183
        s3[i] = c - h_range * 0.275
        s4[i] = c - h_range * 0.55
    
    return r4, r3, r2, r1, s1, s2, s3, s4

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging (mean reversion works)
    CHOP < 38.2 = trending (trend following works)
    """
    n = len(close)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        hi_range = 0.0
        lo_range = 0.0
        for j in range(i - period + 1, i + 1):
            hi_range += high[j] - low[j]
        lo_range = high[i] - low[i]
        
        if lo_range > 0:
            chop[i] = 100 * np.log10(hi_range / lo_range) / np.log10(period)
    
    return chop

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
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA for trend confirmation
    sma_1d = df_1d['close'].rolling(window=21, min_periods=21).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Calculate local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Camarilla pivots (24 periods = 4 days on 4h)
    r4, r3, r2, r1, s1, s2, s3, s4 = calculate_camarilla(high, low, close, period=24)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
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
    cooldown_bars = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(r3[i]) or np.isnan(s3[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Decrease cooldown
        if cooldown_bars > 0:
            cooldown_bars -= 1
        
        # === REGIME CHECK (CHOPPINESS) ===
        # CHOP > 61.8 = ranging, good for mean-reversion at pivots
        # CHOP < 38.2 = trending, avoid (use trend-follow instead)
        chop_val = chop[i] if not np.isnan(chop[i]) else 50.0
        is_ranging = chop_val > 55.0  # Allow more ranging conditions
        is_trending = chop_val < 40.0
        
        # === TREND CHECK (1d SMA) ===
        price_above_1d = close[i] > sma_1d_aligned[i]
        price_below_1d = close[i] < sma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.2
        
        # === DISTANCE TO PIVOTS (for mean-reversion entry) ===
        # Price approaches S3 or R3 → potential reversal
        dist_to_s3 = (close[i] - s3[i]) / atr_14[i]
        dist_to_r3 = (r3[i] - close[i]) / atr_14[i]
        
        # Price bounces off S3 (going up)
        s3_touch = close[i] <= s3[i] * 1.002 and close[i-1] > s3[i] * 1.002 if i > 0 else False
        # Price bounces off R3 (going down)
        r3_touch = close[i] >= r3[i] * 0.998 and close[i-1] < r3[i] * 0.998 if i > 0 else False
        
        # Also check S2/R2 touches (more conservative)
        s2_touch = close[i] <= s2[i] * 1.001 and close[i-1] > s2[i] * 1.001 if i > 0 else False
        r2_touch = close[i] >= r2[i] * 0.999 and close[i-1] < r2[i] * 0.999 if i > 0 else False
        
        desired_signal = 0.0
        
        if not in_position and cooldown_bars == 0:
            # === LONG ENTRY: Price bounces at S3 or S2 support ===
            # Conditions: ranging OR (trending AND at support), volume spike, bullish 1d
            long_trigger = (s3_touch or s2_touch) and vol_spike
            
            if long_trigger and (is_ranging or price_above_1d):
                desired_signal = SIZE
                entry_atr = atr_14[i]
            
            # === SHORT ENTRY: Price bounces at R3 or R2 resistance ===
            # Conditions: ranging OR (trending AND at resistance), volume spike, bearish 1d
            short_trigger = (r3_touch or r2_touch) and vol_spike
            
            if short_trigger and (is_ranging or price_below_1d):
                desired_signal = -SIZE
                entry_atr = atr_14[i]
        
        # === STOPLOSS (2.5 ATR) ===
        if in_position and position_side > 0:
            stop_price = entry_price - 2.5 * entry_atr
            if low[i] < stop_price:
                desired_signal = 0.0
                cooldown_bars = 6  # Cool 24h (6 bars of 4h)
        
        if in_position and position_side < 0:
            stop_price = entry_price + 2.5 * entry_atr
            if high[i] > stop_price:
                desired_signal = 0.0
                cooldown_bars = 6
        
        # === TAKE PROFIT / TRAILING STOP ===
        if in_position and position_side > 0:
            # Trail stop: move up with price
            trailing_stop = close[i] - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            # Take profit at R3 level
            if r3[i] > 0 and close[i] >= r3[i]:
                desired_signal = SIZE / 2  # Half position
                trailing_stop = close[i] - 1.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
                cooldown_bars = 6
        
        if in_position and position_side < 0:
            # Trail stop: move down with price
            trailing_stop = close[i] + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            # Take profit at S3 level
            if s3[i] > 0 and close[i] <= s3[i]:
                desired_signal = -SIZE / 2  # Half position
                trailing_stop = close[i] + 1.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
                cooldown_bars = 6
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                stop_price = close[i] + position_side * (-2.5 * entry_atr)
            elif np.sign(desired_signal) != position_side:
                # Flip position
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                stop_price = close[i] + position_side * (-2.5 * entry_atr)
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals