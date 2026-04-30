#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combination with volume confirmation
# Williams Alligator (JAW/TEETH/LIPS) identifies trendless vs trending markets.
# ADX > 25 confirms trend strength. Go long when LIPS > TEETH > JAW (bullish alignment) and ADX rising.
# Go short when LIPS < TEETH < JAW (bearish alignment) and ADX rising.
# Volume spike confirms participation. Works in bull via trend-following longs, in bear via trend-following shorts.
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ADX_WilliamsAlligator_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Williams Alligator (SMMA = smoothed moving average)
    # JAW: 13-period SMMA, shifted 8 bars forward
    # TEETH: 8-period SMMA, shifted 5 bars forward  
    # LIPS: 5-period SMMA, shifted 3 bars forward
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Shift forward: JAW+8, TEETH+5, LIPS+3
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Calculate ADX (Average Directional Index)
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX, +DI, -DI"""
        if len(high) < period + 1:
            return np.full_like(high, np.nan), np.full_like(high, np.nan), np.full_like(high, np.nan)
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed averages
        def wilders_smoothing(data, period):
            """Wilder's smoothing (EMA with alpha=1/period)"""
            if len(data) < period:
                return np.full_like(data, np.nan)
            result = np.full_like(data, np.nan, dtype=float)
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Wilder's smoothing: alpha = 1/period
            alpha = 1.0 / period
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        tr_smoothed = wilders_smoothing(tr, period)
        plus_dm_smoothed = wilders_smoothing(plus_dm, period)
        minus_dm_smoothed = wilders_smoothing(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(tr_smoothed != 0, (plus_dm_smoothed / tr_smoothed) * 100, 0)
        minus_di = np.where(tr_smoothed != 0, (minus_dm_smoothed / tr_smoothed) * 100, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        adx = wilders_smoothing(dx, period)
        
        return adx, plus_di, minus_di
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for Alligator shifts and ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        curr_adx = adx[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require ADX > 25 (trending market) and volume spike
            if curr_adx > 25.0 and curr_volume_spike:
                # Bullish alignment: LIPS > TEETH > JAW
                if (curr_lips > curr_teeth > curr_jaw):
                    signals[i] = 0.25
                    position = 1
                # Bearish alignment: LIPS < TEETH < JAW
                elif (curr_lips < curr_teeth < curr_jaw):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Alligator alignment breaks (LIPS <= TEETH) or ADX weakens
            if (curr_lips <= curr_teeth) or (curr_adx < 20.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Alligator alignment breaks (LIPS >= TEETH) or ADX weakens
            if (curr_lips >= curr_teeth) or (curr_adx < 20.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals