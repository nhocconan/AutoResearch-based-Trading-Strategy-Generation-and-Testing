#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversals with daily trend filter and volume confirmation.
# Uses daily Camarilla levels (R3/S3 for reversals, R4/S4 for breakouts) and 6h price action.
# In ranging markets: fade extreme touches of R3/S3 with confirmation.
# In trending markets: breakout continuation beyond R4/S4 with volume.
# Works in both bull and bear markets by adapting to regime via price position vs daily VWAP.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13571_6h_camarilla1d_vwap_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
VWAP_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_vwap(high, low, close, volume, period):
    """Calculate VWAP using typical price"""
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).rolling(window=period, min_periods=period).sum()
    vwap_den = volume.rolling(window=period, min_periods=period).sum()
    vwap = vwap_num / vwap_den
    return vwap.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low) * 1.1/2
    # R3 = close + 1.0 * (high - low) * 1.1/2
    # S3 = close - 1.0 * (high - low) * 1.1/2
    # S4 = close - 1.5 * (high - low) * 1.1/2
    range_1d = high_1d - low_1d
    camarilla_multiplier = CAMARILLA_MULT / 2.0  # 1.1/2 = 0.55
    
    r4_1d = close_1d + 1.5 * range_1d * camarilla_multiplier
    r3_1d = close_1d + 1.0 * range_1d * camarilla_multiplier
    s3_1d = close_1d - 1.0 * range_1d * camarilla_multiplier
    s4_1d = close_1d - 1.5 * range_1d * camarilla_multiplier
    
    # Align Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP for trend filter
    vwap = calculate_vwap(high, low, close, volume, VWAP_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VWAP_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vwap[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below VWAP
        above_vwap = close[i] > vwap[i]
        below_vwap = close[i] < vwap[i]
        
        # Signal logic
        if position == 0:
            # Fade extreme touches of R3/S3 (reversal signals)
            fade_s3 = volume_ok and below_vwap and (low[i] <= s3_1d_aligned[i]) and (close[i] > s3_1d_aligned[i])
            fade_r3 = volume_ok and above_vwap and (high[i] >= r3_1d_aligned[i]) and (close[i] < r3_1d_aligned[i])
            
            # Breakout continuation signals (break R4/S4)
            breakout_up = volume_ok and above_vwap and (high[i] > r4_1d_aligned[i])
            breakout_down = volume_ok and below_vwap and (low[i] < s4_1d_aligned[i])
            
            if fade_s3:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_r3:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals