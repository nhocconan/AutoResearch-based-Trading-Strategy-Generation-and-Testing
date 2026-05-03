#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band breakout with 1d ADX trend filter and volume confirmation.
# In trending regime (1d ADX > 25), go long on breakout above upper BB with volume spike.
# In ranging regime (1d ADX < 20), go short on touch of upper BB with volume spike (mean reversion).
# Uses 12h Bollinger Bands (20,2) for structure, 1d ADX for regime filter, and 12h volume spike for confirmation.
# Designed for 50-150 total trades over 4 years with discrete position sizing to minimize fee drag.

name = "12h_Bollinger_Breakout_ADXRegime_VolumeSpike"
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    plus_di_14 = 100 * wilder_smooth(plus_dm, 14) / wilder_smooth(tr, 14)
    minus_di_14 = 100 * wilder_smooth(minus_dm, 14) / wilder_smooth(tr, 14)
    dx = np.where((plus_di_14 + minus_di_14) > 0, 
                  100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14), 
                  0)
    adx_14 = wilder_smooth(dx, 14)
    
    # Align 1d ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 12h Bollinger Bands (20,2)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Calculate volume regime: current 12h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        bb_mid = bb_middle[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(bb_up) or np.isnan(bb_low) or np.isnan(bb_mid) or np.isnan(adx_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: trending if ADX > 25, ranging if ADX < 20
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Generate signals based on regime
        if position == 0:
            if is_trending and close_val > bb_up and vol_spike:
                # Long breakout in trending regime
                signals[i] = 0.25
                position = 1
            elif is_ranging and close_val > bb_up and vol_spike:
                # Short mean reversion in ranging regime (fade the breakout)
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below middle band or regime shifts to ranging
            if close_val < bb_mid or (is_ranging and not is_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above middle band or regime shifts to trending
            if close_val > bb_mid or (is_trending and not is_ranging):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals