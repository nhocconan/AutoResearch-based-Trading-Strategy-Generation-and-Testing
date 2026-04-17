#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme with 1d Volume Spike and ADX Regime Filter.
Long when Williams %R < -80 (oversold) AND 1d volume > 1.5x 20-period average AND ADX < 25 (range/low trend).
Short when Williams %R > -20 (overbought) AND 1d volume > 1.5x 20-period average AND ADX < 25.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) OR ADX > 30 (strong trend, avoid whipsaw).
Uses 1d for volume spike and ADX, 4h for Williams %R. Target: 75-200 total trades over 4 years (19-50/year).
Williams %R captures reversals in bear market rallies/panic dips, volume spike confirms participation, 
ADX filter avoids trading in strong trends where mean reversion fails.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for volume spike and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ADX (14-period)
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
    
    # Smooth TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value: simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / np.where(atr != 0, atr, np.nan)
    minus_di = 100 * minus_dm_smooth / np.where(atr != 0, atr, np.nan)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.nan)
    adx = wilders_smoothing(dx, 14)
    
    # 1d volume spike: volume > 1.5x 20-period EMA
    volume_ema = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * volume_ema)
    
    # Calculate 4h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / np.where((highest_high - lowest_low) != 0, (highest_high - lowest_low), np.nan)
    
    # Align 1d indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # boolean
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND volume spike AND ADX < 25 (low trend)
            if wr < -80 and vol_spike and adx_val < 25:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND volume spike AND ADX < 25
            elif wr > -20 and vol_spike and adx_val < 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR ADX > 30 (strong trend)
            if wr > -50 or adx_val > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR ADX > 30
            if wr < -50 or adx_val > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0