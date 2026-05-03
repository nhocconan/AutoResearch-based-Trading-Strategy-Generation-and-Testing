#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings (>80 or <20) with
# strong trend (ADX>25) and volume spike provide high-probability mean-reversion entries.
# Designed for low trade frequency (12-37/year) on 6h timeframe to minimize fee drag.
# Works in both bull and bear markets by fading extremes in the direction of the 1d trend.

name = "6h_WilliamsR_Extreme_1dADX_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Williams %R, ADX, and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Calculate 1d ADX (14-period)
    plus_dm = pd.Series(df_1d['high'].values).diff()
    minus_dm = pd.Series(df_1d['low'].values).diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr = np.maximum(
        np.maximum(
            pd.Series(df_1d['high'].values) - pd.Series(df_1d['low'].values),
            np.abs(pd.Series(df_1d['high'].values) - pd.Series(df_1d['close'].values).shift(1))
        ),
        np.abs(pd.Series(df_1d['low'].values) - pd.Series(df_1d['close'].values).shift(1))
    )
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with ADX>25 and volume spike
            if williams_r_aligned[i] < -80 and adx_aligned[i] > 25 and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with ADX>25 and volume spike
            elif williams_r_aligned[i] > -20 and adx_aligned[i] > 25 and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (exit oversold) or ADX weakens
            if williams_r_aligned[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (exit overbought) or ADX weakens
            if williams_r_aligned[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals