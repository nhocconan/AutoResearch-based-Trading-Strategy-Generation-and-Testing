#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1d ADX regime filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings (<-90 or >-10) with
# volume spike provide high-probability reversals. ADX > 25 ensures we trade only in trending
# markets to avoid whipsaws in ranging conditions. Designed for low trade frequency (12-37/year)
# on 6h timeframe to minimize fee drag. Works in both bull and bear markets by fading extremes
# in the direction of the higher timeframe trend.

name = "6h_WilliamsR_Extreme_1dADX25_VolumeSpike_Regime"
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
    
    # Get 1d data for Williams %R, ADX, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max()
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high_14 - df_1d['close']) / (highest_high_14 - lowest_low_14)
    williams_r = williams_r.values
    
    # Calculate 1d ADX (14-period)
    plus_dm = pd.Series(df_1d['high'].values).diff()
    minus_dm = pd.Series(df_1d['low'].values).diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = pd.Series(df_1d['high'].values) - pd.Series(df_1d['low'].values)
    tr2 = abs(pd.Series(df_1d['high'].values) - pd.Series(df_1d['close'].values).shift(1))
    tr3 = abs(pd.Series(df_1d['low'].values) - pd.Series(df_1d['close'].values).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
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
        
        # ADX regime filter: only trade when ADX > 25 (trending market)
        is_trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Williams %R extreme oversold (<-90) with volume spike in trending market
            if williams_r_aligned[i] < -90 and volume_spike_aligned[i] and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R extreme overbought (>-10) with volume spike in trending market
            elif williams_r_aligned[i] > -10 and volume_spike_aligned[i] and is_trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 (exiting oversold territory)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 (exiting overbought territory)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals