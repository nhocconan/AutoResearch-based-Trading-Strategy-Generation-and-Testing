#!/usr/bin/env python3
"""
Experiment #028: 4h Camarilla Pivot + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla pivot levels (S3/S4 for longs, R3/R4 for shorts) are 
mathematically precise support/resistance derived from yesterday's range.
Unlike Donchian breakouts, pivot touches capture mean-reversion reversals 
at key levels. Volume spike confirms institutional significance. Choppiness
Index keeps us out of ranging markets where pivots fail.

WHY IT WORKS IN BULL AND BEAR:
- S3/S4: In bull markets, price often bounces from daily support zones
- R3/R4: In bear markets, rallies fail at daily resistance
- Symmetrical: both long and short setups
- 4h TF balances trade frequency with signal quality

TARGET: 75-200 total trades over 4 years (19-50/year)
Signal size: 0.25-0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_1d_v1"
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
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_camarilla_pivots(high, low, close):
    """Camarilla pivot levels using yesterday's HLC"""
    n = len(close)
    
    # Use previous day's values for pivot calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Set first element to NaN (no previous day)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    range_hl = prev_high - prev_low
    
    # Camarilla formulas
    r4 = prev_close + (range_hl * 1.1 / 2)
    r3 = prev_close + (range_hl * 1.1 / 4)
    r2 = prev_close + (range_hl * 1.1 / 6)
    r1 = prev_close + (range_hl * 1.1 / 12)
    
    s1 = prev_close - (range_hl * 1.1 / 12)
    s2 = prev_close - (range_hl * 1.1 / 6)
    s3 = prev_close - (range_hl * 1.1 / 4)
    s4 = prev_close - (range_hl * 1.1 / 2)
    
    return {
        'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4,
        's1': s1, 's2': s2, 's3': s3, 's4': s4
    }

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction
    sma_50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # 1d EMA21 for faster trend
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    pivots = calculate_camarilla_pivots(high, low, close)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Slightly larger since Camarilla is more precise than Donchian
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough for indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_50_aligned[i]) or np.isnan(ema_21_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        current_close = close[i]
        current_high = high[i]
        current_low = low[i]
        
        # === TREND DIRECTION (1d EMA21 vs SMA50) ===
        bull_trend = ema_21_aligned[i] > sma_50_aligned[i]
        bear_trend = ema_21_aligned[i] < sma_50_aligned[i]
        
        # Price relative to 1d SMA50
        price_above_1d_sma = current_close > sma_50_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 50.0
        
        # Skip if too choppy and not in position
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # Get Camarilla levels for this bar
        s3 = pivots['s3'][i]
        s4 = pivots['s4'][i]
        r3 = pivots['r3'][i]
        r4 = pivots['r4'][i]
        
        # Skip if pivots not valid
        if np.isnan(s3) or np.isnan(r3):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Price touches S3 or S4 support with volume ===
            # S3 is more aggressive, S4 is extreme overshoot
            long_trigger = False
            
            # Touch S3 support (primary)
            if s4 <= current_low <= s3 and bull_trend:
                if vol_spike or is_trending:
                    long_trigger = True
            
            # Touch S4 extreme (more extreme entry)
            if current_low <= s4 and bull_trend:
                if vol_spike:
                    long_trigger = True
            
            # Alternative: price within 0.1% of S3 in strong uptrend
            s3_tolerance = s3 * 0.001
            if abs(current_close - s3) < s3_tolerance and price_above_1d_sma:
                if vol_spike:
                    long_trigger = True
            
            if long_trigger:
                desired_signal = SIZE
            
            # === SHORT ENTRY: Price touches R3 or R4 resistance with volume ===
            short_trigger = False
            
            # Touch R3 resistance (primary)
            if r3 <= current_high <= r4 and bear_trend:
                if vol_spike or is_trending:
                    short_trigger = True
            
            # Touch R4 extreme
            if current_high >= r4 and bear_trend:
                if vol_spike:
                    short_trigger = True
            
            # Alternative: price within 0.1% of R3 in downtrend
            r3_tolerance = r3 * 0.001
            if abs(current_close - r3) < r3_tolerance and not price_above_1d_sma:
                if vol_spike:
                    short_trigger = True
            
            if short_trigger:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Long: stop below entry minus 2*ATR, or below S4
            stoploss_long = min(entry_price - 2.0 * entry_atr, s4 * 0.99)
            if current_low < stoploss_long:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Short: stop above entry plus 2*ATR, or above R4
            stoploss_short = max(entry_price + 2.0 * entry_atr, r4 * 1.01)
            if current_high > stoploss_short:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT (optional: trail stop at mid-levels) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 4:
            # Exit if price reaches opposite pivot level
            if position_side > 0:
                # Long: exit near R1 or R2
                r1 = pivots['r1'][i]
                r2 = pivots['r2'][i]
                if not np.isnan(r1) and current_close >= r1:
                    desired_signal = SIZE / 2  # Half position
                if not np.isnan(r2) and current_close >= r2:
                    desired_signal = 0.0  # Full exit
            else:
                # Short: exit near S1 or S2
                s1 = pivots['s1'][i]
                s2 = pivots['s2'][i]
                if not np.isnan(s1) and current_close <= s1:
                    desired_signal = -SIZE / 2  # Half position
                if not np.isnan(s2) and current_close <= s2:
                    desired_signal = 0.0  # Full exit
        
        # === TIME-BASED EXIT (hold at least 8 bars = 2 days max) ===
        if in_position and bars_held >= 8:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = current_close
                entry_atr = atr_14[i]
                entry_bar = i
                
                # Set initial stop
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals