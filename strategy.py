#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (13/8/5 SMAs) + 1d ADX regime filter + volume confirmation
# Williams Alligator identifies trend direction (JAW=13, TEETH=8, LIPS=5 SMAs)
# 1d ADX > 25 filters for trending markets only (avoids whipsaws in ranges)
# Volume confirmation (1.5x 20-period avg) ensures breakout strength
# Discrete sizing 0.25 to limit fee drift. Target: 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear: Alligator catches trends, ADX avoids false signals in consolidation

name = "12h_1d_alligator_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Williams Alligator (JAW=13, TEETH=8, LIPS=5 SMAs)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # SMAs with proper min_periods handling
    close_series = pd.Series(close)
    jaw = close_series.rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = close_series.rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = close_series.rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Calculate 20-period average volume for volume confirmation
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Alligator reverses (Lips < Teeth) OR ADX < 20 (trend weakening)
            if lips[i] < teeth[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator reverses (Lips > Teeth) OR ADX < 20 (trend weakening)
            if lips[i] > teeth[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Alligator alignment + ADX regime
            if volume_confirmed and adx_aligned[i] > 25:
                # Long entry: Lips > Teeth > Jaw (bullish alignment) + ADX > 25 (strong trend)
                if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Lips < Teeth < Jaw (bearish alignment) + ADX > 25 (strong trend)
                elif lips[i] < teeth[i] and teeth[i] < jaw[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals