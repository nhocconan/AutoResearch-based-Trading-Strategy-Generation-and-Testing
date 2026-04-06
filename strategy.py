#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels from 1D with volume confirmation.
# In bull markets, price breaks above R4 with volume, signaling continuation.
# In bear markets, price breaks below S4 with volume, signaling continuation.
# Fade at R3/S3 when price fails to break through on weak volume.
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost.

name = "exp_13231_6h_camarilla1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1
VOLUME_MA_PERIOD = 20
VOLUME_BREAKOUT_THRESHOLD = 1.5
VOLUME_FADE_THRESHOLD = 0.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    c = close + (range_val * 1.1 / 12)
    l3 = close - (range_val * 1.1 / 6)
    h3 = close + (range_val * 1.1 / 6)
    l4 = close - (range_val * 1.1 / 2)
    h4 = close + (range_val * 1.1 / 2)
    return c, l3, h3, l4, h4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_c = np.full_like(close_1d, np.nan)
    camarilla_l3 = np.full_like(close_1d, np.nan)
    camarilla_h3 = np.full_like(close_1d, np.nan)
    camarilla_l4 = np.full_like(close_1d, np.nan)
    camarilla_h4 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        c, l3, h3, l4, h4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        camarilla_c[i] = c
        camarilla_l3[i] = l3
        camarilla_h3[i] = h3
        camarilla_l4[i] = l4
        camarilla_h4[i] = h4
    
    # Align Camarilla levels to 6H timeframe (shifted by 1 day for completed bar)
    camarilla_c_aligned = align_htf_to_ltf(prices, df_1d, camarilla_c)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    
    # Calculate 6H indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if (np.isnan(camarilla_c_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume conditions
        vol_ma = volume_ma[i]
        vol_ok = not np.isnan(vol_ma)
        volume_high = vol_ok and (volume[i] > (vol_ma * VOLUME_BREAKOUT_THRESHOLD))
        volume_low = vol_ok and (volume[i] < (vol_ma * VOLUME_FADE_THRESHOLD))
        
        # Camarilla levels
        c = camarilla_c_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l4 = camarilla_l4_aligned[i]
        h4 = camarilla_h4_aligned[i]
        
        # Breakout and fade signals
        breakout_up = volume_high and (close[i] > h4)
        breakout_down = volume_high and (close[i] < l4)
        fade_up = volume_low and (close[i] > h3 and close[i] < h4)
        fade_down = volume_low and (close[i] > l3 and close[i] < l4)
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_up:
                signals[i] = -SIGNAL_SIZE * 0.5  # Small short on fade
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_down:
                signals[i] = SIGNAL_SIZE * 0.5  # Small long on fade
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on fade down or reverse signal
            if fade_down or (close[i] < l3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on fade up or reverse signal
            if fade_up or (close[i] > h3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals