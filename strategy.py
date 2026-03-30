#!/usr/bin/env python3
"""
Experiment #005: 12h Camarilla + Choppiness Regime + Volume Spike

HYPOTHESIS: The proven Camarilla+volume+choppiness pattern from 4h (Sharpe 1.47)
transfers to 12h with 1d HTF trend. Different entry logic per regime:
- CHOP < 50 (trending): Enter on S4/R4 breakouts (strong momentum)
- CHOP > 61.8 (choppy): Enter on S3/R3 touches (mean reversion)
- Volume spike required for all entries
- 1d EMA21 for trend direction

WHY 12h: 3x fewer trades than 4h = less fee drag, better generalization.
Target: 75-150 total trades over 4 years.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_chop_vol_1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures trendiness vs choppiness"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    chop = np.full(n, 50.0)
    for i in range(period, n):
        highest_high = max(high[i - period:i + 1])
        lowest_low = min(low[i - period:i + 1])
        
        range_sum = highest_high - lowest_low
        if range_sum <= 0:
            chop[i] = 50.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

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
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100
    
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
        
        # === TREND DIRECTION (1d EMA21) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === REGIME (Choppiness) ===
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 50.0
        
        # === CAMARILLA LEVELS from previous CLOSED bar (no look-ahead) ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        if prev_range <= 0:
            signals[i] = 0.0
            continue
        
        # Classic Camarilla levels (factor 1.1/12 = 0.09167)
        r3 = prev_close + prev_range * 0.09167 * 2  # R3 = C + range * 0.1833
        r4 = prev_close + prev_range * 0.09167 * 4  # R4 = C + range * 0.3667
        s3 = prev_close - prev_range * 0.09167 * 2  # S3 = C - range * 0.1833
        s4 = prev_close - prev_range * 0.09167 * 4  # S4 = C - range * 0.3667
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Trend up + volume + Camarilla S-level touch ===
            if price_above_1d_ema and vol_spike:
                # S4 touch in trending market (strong breakout level)
                if is_trending and low[i] <= s4:
                    desired_signal = SIZE
                # S3 touch in choppy market (mean reversion)
                elif is_choppy and low[i] <= s3:
                    desired_signal = SIZE
            
            # === SHORT: Trend down + volume + Camarilla R-level touch ===
            if not price_above_1d_ema and vol_spike:
                # R4 touch in trending market
                if is_trending and high[i] >= r4:
                    desired_signal = -SIZE
                # R3 touch in choppy market
                elif is_choppy and high[i] >= r3:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD: 2 bars (24h) to reduce fee churn ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 2:
            # Take profit if price returns to prev close
            if position_side > 0 and close[i] >= prev_close:
                desired_signal = 0.0
            if position_side < 0 and close[i] <= prev_close:
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
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals