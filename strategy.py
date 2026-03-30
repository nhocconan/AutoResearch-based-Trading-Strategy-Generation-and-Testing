#!/usr/bin/env python3
"""
Experiment #028: 6h Camarilla Pivot + A/D Momentum + Volume Spike + 1d SMA200

HYPOTHESIS: Camarilla pivot levels from 1d provide clear structure boundaries.
A/D line (Accumulation/Distribution) measures institutional money flow - NOT the
same as RSI/Williams %R that failed repeatedly. Volume spike confirms institutional
participation at key levels. 1d SMA200 filters for bull/bear regime.

WHY 6h: Slower than 4h (fewer whipsaws), faster than 12h (more opportunities).
Camarilla R3/S3 are strong reversal zones, R4/S4 are breakout continuation levels.

WHY A/D IS NOVEL: Previous failures used Williams %R, RSI, TRIX - all momentum
oscillators. A/D measures VOLUME-WEIGHTED money flow, capturing institutional
accumulation/distribution patterns that price-based oscillators miss.

TARGET: 75-150 total trades over 4 years. HARD MAX: 300.
Signal size: 0.25 (conservative).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_ad_vol_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla_pivots(high, low, close, period=1):
    """
    Camarilla Pivot Levels (multiplied for higher timeframe reference)
    R4 = close + (high - low) * 1.1/2
    R3 = close + (high - low) * 1.1/3
    S3 = close - (high - low) * 1.1/3
    S4 = close - (high - low) * 1.1/2
    """
    n = len(close)
    r4 = np.full(n, np.nan, dtype=np.float64)
    r3 = np.full(n, np.nan, dtype=np.float64)
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        prev_high = high[i - period]
        prev_low = low[i - period]
        prev_close = close[i - period]
        
        day_range = prev_high - prev_low
        
        r4[i] = prev_close + day_range * 0.55
        r3[i] = prev_close + day_range * 0.3667
        s3[i] = prev_close - day_range * 0.3667
        s4[i] = prev_close - day_range * 0.55
    
    return r4, r3, s3, s4

def calculate_ad(high, low, close, volume):
    """
    Accumulation/Distribution Line
    Novel momentum indicator measuring institutional money flow
    """
    n = len(close)
    ad = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        high_i = high[i]
        low_i = low[i]
        close_i = close[i]
        close_prev = close[i - 1]
        
        high_low_range = high_i - low_i
        
        if high_low_range > 1e-10:
            # Money flow multiplier
            mfm = ((close_i - low_i) - (high_i - close_i)) / high_low_range
            mfv = mfm * volume[i]
            ad[i] = ad[i - 1] + mfv
        else:
            ad[i] = ad[i - 1]
    
    return ad

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

def calculate_ad_momentum(ad, period=10):
    """A/D momentum: rate of change of money flow"""
    n = len(ad)
    momentum = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        if not np.isnan(ad[i - period]) and not np.isnan(ad[i]):
            momentum[i] = (ad[i] - ad[i - period]) / (abs(ad[i - period]) + 1e-10)
    
    return momentum

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Camarilla pivots from 6h bars (using prior bar for calculation)
    r4, r3, s3, s4 = calculate_camarilla_pivots(high, low, close, period=1)
    
    # A/D line and momentum
    ad = calculate_ad(high, low, close, volume)
    ad_mom = calculate_ad_momentum(ad, period=10)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # A/D EMA for smoothing
    ad_ema = pd.Series(ad).ewm(span=10, min_periods=10, adjust=False).mean().values
    
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
    
    warmup = 250  # Need enough for SMA200(200) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ad_mom[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_200_aligned[i]
        
        # === A/D MOMENTUM (novel indicator) ===
        ad_positive = ad_mom[i] > 0.1  # Significant accumulation
        ad_negative = ad_mom[i] < -0.1  # Significant distribution
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === Camarilla levels ===
        r4_val = r4[i] if not np.isnan(r4[i]) else 0
        r3_val = r3[i] if not np.isnan(r3[i]) else 0
        s3_val = s3[i] if not np.isnan(s3[i]) else 0
        s4_val = s4[i] if not np.isnan(s4[i]) else 0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Price at/below S3 with accumulation ===
            # Fade the reversal at S3 support
            if price_above_1d_sma:  # Uptrend
                if close[i] <= s3_val * 1.01 and close[i] >= s4_val * 0.99:
                    if ad_positive and (vol_spike or ad_mom[i] > 0.2):
                        desired_signal = SIZE
            
            # === SHORT ENTRY: Price at/above R3 with distribution ===
            # Fade the reversal at R3 resistance
            if not price_above_1d_sma:  # Downtrend
                if close[i] >= r3_val * 0.99 and close[i] <= r4_val * 1.01:
                    if ad_negative and (vol_spike or ad_mom[i] < -0.2):
                        desired_signal = -SIZE
            
            # === BREAKOUT CONTINUATION (backup signals) ===
            # Long breakout above R4
            if price_above_1d_sma:
                if close[i] > r4_val and vol_spike and ad_positive:
                    desired_signal = SIZE
            
            # Short breakdown below S4
            if not price_above_1d_sma:
                if close[i] < s4_val and vol_spike and ad_negative:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR trailing) ===
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
        
        # === TIME-BASED EXIT (hold at least 8 bars = 2 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit on reversal of momentum
            if position_side > 0 and ad_negative:
                desired_signal = 0.0
            if position_side < 0 and ad_positive:
                desired_signal = 0.0
        
        # === TAKE PROFIT at 2R ===
        if in_position and position_side > 0:
            profit_target = entry_price + 2.0 * entry_atr
            if high[i] >= profit_target:
                desired_signal = SIZE / 2  # Take half profit
        
        if in_position and position_side < 0:
            profit_target = entry_price - 2.0 * entry_atr
            if low[i] <= profit_target:
                desired_signal = -SIZE / 2  # Take half profit
        
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
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals