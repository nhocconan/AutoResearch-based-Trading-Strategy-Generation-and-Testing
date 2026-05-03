#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume spike + chop regime filter
# Donchian channels provide robust structure for breakouts in both bull and bear markets.
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades.
# Volume confirmation filters false breakouts.
# Chop regime filter avoids trading in sideways markets where breakouts fail.
# Designed for 30-100 total trades over 4 years (7-25/year) on 1d timeframe.

name = "1d_Donchian20_1wEMA50_VolumeSpike_ChopFilter"
timeframe = "1d"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels from previous 1d bar (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 20-period EMA on 1d
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Chop regime filter: 14-period chopiness index
    # Chop > 61.8 = range (avoid), Chop < 38.2 = trending (favor breakouts)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    chop_denom = max_high - min_low
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10(atr * np.sqrt(atr_period) / chop_denom) / np.log10(atr_period)
    chop = np.where(np.isnan(chop), 50.0, chop)  # Default to neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid Donchian and indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(chop[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        chop_filter = chop[i] < 50.0  # Prefer trending markets (chop < 50)
        
        if position == 0:
            # Long: price breaks above upper Donchian in uptrend alignment with volume spike and chop filter
            if close[i] > donchian_upper[i] and ema_50_1w_aligned[i] < close[i] and volume_spike and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian in downtrend alignment with volume spike and chop filter
            elif close[i] < donchian_lower[i] and ema_50_1w_aligned[i] > close[i] and volume_spike and chop_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian or loses uptrend alignment
            if close[i] < donchian_lower[i] or ema_50_1w_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian or loses downtrend alignment
            if close[i] > donchian_upper[i] or ema_50_1w_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals