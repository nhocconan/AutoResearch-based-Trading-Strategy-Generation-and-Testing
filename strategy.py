#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1d chop regime filter
    # Donchian breakout captures momentum, volume confirms strength, chop filter avoids false signals in ranging markets
    # Works in bull/bear by only taking breakouts aligned with higher timeframe structure
    # Target: 20-50 trades/year per symbol (80-200 total over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for chop filter
    tr_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i == 0:
            tr_1d[i] = high_1d[i] - low_1d[i]
        else:
            tr_1d[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if i == 13:
            atr_1d[i] = np.mean(tr_1d[i-13:i+1])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate 1d ADX(14) for trend strength (alternative to chop)
    # +DM, -DM
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
    
    # Smoothed +DM, -DM, TR
    tr_14 = np.full(len(df_1d), np.nan)
    plus_dm_14 = np.full(len(df_1d), np.nan)
    minus_dm_14 = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if i == 13:
            tr_14[i] = np.mean(tr_1d[i-13:i+1])
            plus_dm_14[i] = np.mean(plus_dm[i-13:i+1])
            minus_dm_14[i] = np.mean(minus_dm[i-13:i+1])
        else:
            tr_14[i] = (tr_14[i-1] * 13 + tr_1d[i]) / 14
            plus_dm_14[i] = (plus_dm_14[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_14[i] = (minus_dm_14[i-1] * 13 + minus_dm[i]) / 14
    
    # +DI, -DI, DX, ADX
    plus_di = np.full(len(df_1d), np.nan)
    minus_di = np.full(len(df_1d), np.nan)
    dx = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if tr_14[i] != 0:
            plus_di[i] = (plus_dm_14[i] / tr_14[i]) * 100
            minus_di[i] = (minus_dm_14[i] / tr_14[i]) * 100
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    adx = np.full(len(df_1d), np.nan)
    for i in range(26, len(df_1d)):
        if i == 26:
            adx[i] = np.mean(dx[i-13:i+1])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Calculate 1d volume MA(20) for volume spike filter
    vol_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        if i == 19:
            vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
        else:
            vol_ma_20[i] = (vol_ma_20[i-1] * 19 + volume_1d[i]) / 20
    
    # Align 1d indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current 4h volume > 1.5x 1d average volume (scaled)
        # Approximate 1d volume per 4h bar: 1d volume / 6 (since 6x 4h in 1d)
        vol_4h = volume[i]
        vol_1d_per_4h = vol_ma_20_aligned[i] / 6.0
        volume_spike = vol_4h > 1.5 * vol_1d_per_4h
        
        # Trend filter: ADX > 25 indicates strong trend (avoid chop)
        strong_trend = adx_1d_aligned[i] > 25
        
        # Donchian breakout conditions
        bullish_breakout = close[i] > highest_high[i-1]  # Break above previous high
        bearish_breakout = close[i] < lowest_low[i-1]    # Break below previous low
        
        # Entry logic
        long_entry = bullish_breakout and volume_spike and strong_trend
        short_entry = bearish_breakout and volume_spike and strong_trend
        
        # Exit logic: opposite breakout or loss of momentum
        long_exit = bearish_breakout or (adx_1d_aligned[i] < 20)
        short_exit = bullish_breakout or (adx_1d_aligned[i] < 20)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0