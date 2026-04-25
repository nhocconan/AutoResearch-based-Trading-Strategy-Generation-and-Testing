#!/usr/bin/env python3
"""
1h Volume Spike + 4h Donchian Breakout + 1d EMA50 Trend
Hypothesis: On 1h, volume spikes confirm institutional interest. 4h Donchian breakout provides directional bias,
filtered by 1d EMA50 trend to avoid counter-trend trades. Session filter (08-20 UTC) reduces noise.
Designed for low trade frequency (15-35/year) to minimize fee drag in ranging/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) once
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h Donchian: upper = max(high,20), lower = min(low,20)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d EMA50 for trend
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss on 1h
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup for indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        atr_val = atr[i]
        
        # Volume spike: current volume > 2.5 * 24-period average (stricter for 1h)
        if i >= 24:
            vol_ma_24 = np.mean(volume[i-23:i+1])
        else:
            vol_ma_24 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.5 * vol_ma_24
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50
        downtrend = curr_close < ema_50
        
        if position == 0:
            # Long: price breaks above 4h Donchian high AND volume spike AND uptrend
            long_condition = (curr_high > donchian_high_val) and volume_spike and uptrend
            # Short: price breaks below 4h Donchian low AND volume spike AND downtrend
            short_condition = (curr_low < donchian_low_val) and volume_spike and downtrend
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or trend reversal
            if curr_close <= entry_price - 2.0 * atr_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or trend reversal
            if curr_close >= entry_price + 2.0 * atr_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_DonchianBreakout_1dEMA50_Trend_v1"
timeframe = "1h"
leverage = 1.0