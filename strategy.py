#!/usr/bin/env python3
"""
Experiment #7767: 6-hour 1-day Camarilla pivot + volume confirmation with trend filter.
Hypothesis: Fade at Camarilla R3/S3 levels during ranging markets and breakout continuation at R4/S4 during trending markets, with 1d ADX filter to distinguish regimes. Works in bull markets (buy R3 bounce, sell R4 breakout) and bear markets (sell S3 bounce, buy S4 breakdown). Targets 50-150 trades over 4 years.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7767_6h_camarilla_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25  # ADX > 25 = trending
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * CAMARILLA_MULT * 1.1)
    s3 = pivot - (range_1d * CAMARILLA_MULT * 1.1)
    r4 = pivot + (range_1d * CAMARILLA_MULT * 1.5)
    s4 = pivot - (range_1d * CAMARILLA_MULT * 1.5)
    
    # Align Camarilla levels to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d ADX for trend filter
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d.values).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Directional indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(adx_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
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
        
        # Determine market regime
        trending = adx_aligned[i] > ADX_TREND_THRESHOLD
        ranging = adx_aligned[i] <= ADX_TREND_THRESHOLD
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Price levels
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        r4_level = r4_aligned[i]
        s4_level = s4_aligned[i]
        
        # Entry conditions based on regime
        if ranging:
            # Fade at S3/R3 in ranging markets
            long_entry = (close[i] > s3_level) and (close[i-1] <= s3_level) and volume_confirmed
            short_entry = (close[i] < r3_level) and (close[i-1] >= r3_level) and volume_confirmed
        else:
            # Breakout continuation at S4/R4 in trending markets
            long_entry = (close[i] > r4_level) and volume_confirmed
            short_entry = (close[i] < s4_level) and volume_confirmed
        
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