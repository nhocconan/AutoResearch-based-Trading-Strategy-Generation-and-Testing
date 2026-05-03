#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume spike and ADX regime filter
# Donchian breakouts capture momentum; volume confirms institutional participation;
# ADX > 25 ensures trending market (avoids chop). Works in bull/bear by trading
# breakouts in direction of higher timeframe EMA trend. Target: 20-50 trades/year.

name = "4h_Donchian20_VolumeSpike_ADXTrend"
timeframe = "4h"
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
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Calculate 1d ADX(14) for regime filter
    # ADX requires TR, +DM, -DM
    tr1 = pd.Series(df_1d['high']).rolling(1).max() - pd.Series(df_1d['low']).rolling(1).min()
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr_rma = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_dm_rma = plus_dm.ewm(alpha=1/14, adjust=False).mean()
    minus_dm_rma = minus_dm.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm_rma / tr_rma.replace(0, np.nan))
    minus_di = 100 * (minus_dm_rma / tr_rma.replace(0, np.nan))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    adx_trend = adx_values > 25  # Trending regime
    
    # Align 1d indicators to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    adx_trend_aligned = align_htf_to_ltf(prices, df_1d, adx_trend)
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback = 20
    upper_channel = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_channel = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(adx_trend_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper channel with volume spike in uptrend + ADX>25
            if close[i] > upper_channel[i] and close[i-1] <= upper_channel[i-1] and ema_50_aligned[i] > close[i] and volume_spike_aligned[i] and adx_trend_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel with volume spike in downtrend + ADX>25
            elif close[i] < lower_channel[i] and close[i-1] >= lower_channel[i-1] and ema_50_aligned[i] < close[i] and volume_spike_aligned[i] and adx_trend_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters below upper channel
            if close[i] < upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters above lower channel
            if close[i] > lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals