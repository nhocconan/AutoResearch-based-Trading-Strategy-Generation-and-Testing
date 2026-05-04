#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ADX regime filter
# Donchian breakouts capture strong momentum moves. Volume confirmation (2.0x 20-period EMA)
# ensures breakout validity. 1d ADX > 25 filters for trending regimes, avoiding false breakouts
# in ranging markets. Designed for 4h timeframe to target 20-50 trades/year (75-200 total over 4 years)
# with discrete sizing (0.30). Works in bull markets by buying breakouts above upper channel in
# uptrends and in bear markets by selling breakdowns below lower channel in downtrends.

name = "4h_Donchian20_Volume_1dADX25_Trend"
timeframe = "4h"
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
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = WilderSmooth(tr, 14)
    plus_di_1d = 100 * WilderSmooth(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * WilderSmooth(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = WilderSmooth(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channels on 4h
    lookback = 20
    upper_channel = np.maximum.accumulate(high)
    upper_channel = np.where(np.arange(len(high)) < lookback-1, np.nan, upper_channel)
    for i in range(lookback-1, len(high)):
        upper_channel[i] = np.max(high[i-lookback+1:i+1])
    
    lower_channel = np.minimum.accumulate(low)
    lower_channel = np.where(np.arange(len(low)) < lookback-1, np.nan, lower_channel)
    for i in range(lookback-1, len(low)):
        lower_channel[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: 2.0x 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20[i])
        
        # Regime filter: 1d ADX > 25 indicates trending market
        trending_market = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: close breaks above upper Donchian + volume confirmation + trending market
            if (close[i] > upper_channel[i] and volume_confirmed and trending_market):
                signals[i] = 0.30
                position = 1
            # Short: close breaks below lower Donchian + volume confirmation + trending market
            elif (close[i] < lower_channel[i] and volume_confirmed and trending_market):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price falls below lower Donchian (mean reversion) OR ADX drops below 20 (range)
            if close[i] < lower_channel[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above upper Donchian (mean reversion) OR ADX drops below 20 (range)
            if close[i] > upper_channel[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals