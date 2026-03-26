#!/usr/bin/env python3
"""
Experiment #027: 12h Camarilla S3/R3 + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla pivot levels (S3/R3) are mathematically derived support/resistance
zones used by institutional traders. Price tends to bounce at these levels. Combined with
volume spike (institutional activity) and Choppiness filter (avoid ranging markets),
this captures mean-reversion at key turning points.

WHY 12h: 12-25 trades/year = enough for statistics, slow enough to avoid fee drag.
Camarilla levels on 12h form meaningful zones without noise of lower timeframes.

SIMPLICITY: ONE price structure (Camarilla) + volume confirmation + regime filter.
DB winner pattern: mtf_4h_camarilla_pivot_volume_spike_choppiness (ETH test Sharpe 1.47)

TARGET: 50-150 total over 4 years (12-37/year). HARD MAX: 200.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_chop_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP < 38.2 = trending (momentum works)
    CHOP > 61.8 = choppy/range (mean reversion at Camarilla levels works)
    """
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_camarilla_levels(high, low, close):
    """
    Camarilla Pivot Levels (simplified S3/R3)
    S3 = low + 2 * (pivot - low)
    S4 = low + (high - low) * 1.1 / 6
    R3 = high - 2 * (high - pivot)
    R4 = high - (high - low) * 1.1 / 6
    where pivot = (high + low + close) / 3
    """
    n = len(close)
    pivot = (high + low + close) / 3.0
    
    s3 = low + 2.0 * (pivot - low)
    s4 = low + (high - low) * 1.1 / 6.0
    r3 = high - 2.0 * (high - pivot)
    r4 = high - (high - low) * 1.1 / 6.0
    
    return s3, s4, r3, r4, pivot

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR for stoploss sizing
    atr_1d_raw = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_raw)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    s3, s4, r3, r4, pivot = calculate_camarilla_levels(high, low, close)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # 12h is slower, less warmup needed
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME (Choppiness Index) ===
        # CHOP > 61.8 = choppy (mean reversion at Camarilla works)
        # CHOP < 38.2 = trending (momentum breakout works better)
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8  # Strict volume filter
        
        # === CAMARILLA LEVEL PROXIMITY ===
        # Price within 0.3% of level = touching it
        tolerance = 0.003
        
        near_s3 = abs(close[i] - s3[i]) / close[i] < tolerance
        near_s4 = abs(close[i] - s4[i]) / close[i] < tolerance
        near_r3 = abs(close[i] - r3[i]) / close[i] < tolerance
        near_r4 = abs(close[i] - r4[i]) / close[i] < tolerance
        
        # Price below pivot = potential bounce up
        price_below_pivot = close[i] < pivot[i]
        # Price above pivot = potential bounce down
        price_above_pivot = close[i] > pivot[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Price at S3/S4 in choppy market ===
            if is_choppy and vol_spike:
                # Bounce from S3
                if near_s3 and price_below_pivot:
                    # Check if price is recovering (close > low of current bar)
                    if close[i] > low[i] + atr_14[i] * 0.5:
                        desired_signal = SIZE
                
                # Bounce from S4 (stronger support)
                if near_s4 and price_below_pivot:
                    if close[i] > low[i] + atr_14[i] * 0.5:
                        desired_signal = SIZE
            
            # === SHORT ENTRY: Price at R3/R4 in choppy market ===
            if is_choppy and vol_spike:
                # Rejection from R3
                if near_r3 and price_above_pivot:
                    # Price falling from R3
                    if close[i] < high[i] - atr_14[i] * 0.5:
                        desired_signal = -SIZE
                
                # Rejection from R4
                if near_r4 and price_above_pivot:
                    if close[i] < high[i] - atr_14[i] * 0.5:
                        desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR trailing) ===
        stoploss_triggered = False
        stop_price = 0.0
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            if low[i] < trailing_stop:
                stoploss_triggered = True
                stop_price = trailing_stop
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            if high[i] > trailing_stop:
                stoploss_triggered = True
                stop_price = trailing_stop
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT (2:1 R:R) ===
        if in_position and not stoploss_triggered:
            bars_held = i - entry_bar
            min_hold_bars = 2  # At least 1 day hold
            
            if bars_held >= min_hold_bars:
                if position_side > 0:
                    profit_pct = (close[i] - entry_price) / entry_price
                    risk_pct = 2.0 * entry_atr / entry_price
                    # Take profit at 2:1
                    if profit_pct >= 2.0 * risk_pct:
                        desired_signal = 0.0
                    # Also exit if hits R3/R4
                    if near_r3 or near_r4:
                        desired_signal = 0.0
                
                if position_side < 0:
                    profit_pct = (entry_price - close[i]) / entry_price
                    risk_pct = 2.0 * entry_atr / entry_price
                    # Take profit at 2:1
                    if profit_pct >= 2.0 * risk_pct:
                        desired_signal = 0.0
                    # Also exit if hits S3/S4
                    if near_s3 or near_s4:
                        desired_signal = 0.0
        
        # === RSI EXIT FILTER ===
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = (100 - (100 / (1 + rs)))[i]
        
        if in_position:
            if position_side > 0 and rsi > 75:
                desired_signal = 0.0
            if position_side < 0 and rsi < 25:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
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
            # else: maintain position
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals