#!/usr/bin/env python3
"""
Experiment #8439: 6h Camarilla pivot + volume spike + regime filter.
Hypothesis: Camarilla pivot levels provide institutional support/resistance zones.
Price breaking above R4 or below S4 with volume spike indicates institutional breakout.
Price rejecting at R3/S3 with volume spike indicates institutional fade.
Using 12h ADX regime filter: ADX>25 for trend following (breakouts), ADX<20 for mean reversion (fades).
Targets 75-200 total trades over 4 years (19-50/year) to balance opportunity with fee drag.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8439_6h_camarilla12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
VOLUME_LOOKBACK = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+
    tr_period = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    dm_plus_period = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    dm_minus_period = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    return adx.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla levels (based on previous day's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels for 12h timeframe
    range_12h = high_12h - low_12h
    camarilla_h5 = close_12h + (range_12h * 1.1 / 2)  # H5 = Close + 1.1*(Range)/2
    camarilla_h4 = close_12h + (range_12h * 1.1)      # H4 = Close + 1.1*Range
    camarilla_h3 = close_12h + (range_12h * 1.1/2) * 2  # H3 = Close + 1.1*Range
    camarilla_l3 = close_12h - (range_12h * 1.1/2) * 2  # L3 = Close - 1.1*Range
    camarilla_l4 = close_12h - (range_12h * 1.1)      # L4 = Close - 1.1*Range
    camarilla_l5 = close_12h - (range_12h * 1.1 / 2)  # L5 = Close - 1.1*(Range)/2
    
    # Align Camarilla levels to 6h timeframe
    h5_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h5)
    h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    l5_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l5)
    
    # Calculate 12h ADX for regime filter
    adx_12h = calculate_adx(df_12h['high'], df_12h['low'], df_12h['close'], ADX_PERIOD)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CAMARILLA_PERIOD, ADX_PERIOD, VOLUME_LOOKBACK, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(adx_aligned[i]) or np.isnan(h4_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine regime from 12h ADX
        trending = adx_aligned[i] > ADX_TREND_THRESHOLD   # ADX > 25: trending
        ranging = adx_aligned[i] < ADX_RANGE_THRESHOLD    # ADX < 20: ranging
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Camarilla-based signals
        # Breakout signals (trending regime): break H4/L4 with volume
        breakout_long = trending and close[i] > h4_aligned[i-1] and volume_confirmed
        breakout_short = trending and close[i] < l4_aligned[i-1] and volume_confirmed
        
        # Fade signals (ranging regime): reject at H3/L3 with volume
        fade_long = ranging and close[i] < h3_aligned[i-1] and close[i] > l3_aligned[i-1] and \
                    volume[i] > volume[i-1] and volume_confirmed  # Price rejecting from upper range
        fade_short = ranging and close[i] > l3_aligned[i-1] and close[i] < h3_aligned[i-1] and \
                     volume[i] > volume[i-1] and volume_confirmed  # Price rejecting from lower range
        
        # Entry conditions
        long_entry = breakout_long or fade_long
        short_entry = breakout_short or fade_short
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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