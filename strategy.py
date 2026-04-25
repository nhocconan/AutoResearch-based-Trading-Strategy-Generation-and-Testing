#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w EMA50 Trend + Volume Spike + ATR Stoploss
Hypothesis: Daily Donchian breakouts capture significant multi-day moves. The 1w EMA50 provides a robust trend filter that works in both bull and bear markets (avoiding counter-trend trades). Volume spike confirms breakout authenticity. ATR-based stoploss manages risk. Daily timeframe minimizes fee drag while capturing major swings. Works in bull markets via trend-following breaks and in bear markets via short breaks below Donchian lower band in downtrends.
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
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 1d ATR(14) for stoploss
    tr1 = pd.Series(df_1d['high']).shift(1) - pd.Series(df_1d['low'])
    tr2 = abs(pd.Series(df_1d['high']).shift(1) - pd.Series(df_1d['close']))
    tr3 = abs(pd.Series(df_1d['low']).shift(1) - pd.Series(df_1d['close']))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for Donchian warmup and EMA50 alignment
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        atr_val = atr_aligned[i]
        ema_trend = ema_50_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND above 1w EMA50 (uptrend filter)
            long_condition = (curr_close > donchian_high) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below Donchian lower band AND below 1w EMA50 (downtrend filter)
            short_condition = (curr_close < donchian_low) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - 2.5 * atr_val  # 2.5 ATR stoploss
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + 2.5 * atr_val  # 2.5 ATR stoploss
        elif position == 1:
            # Long position management
            # Stoploss: price closes below ATR stop level
            if curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            # Exit: price returns below Donchian lower band or trend breaks
            elif curr_close <= donchian_low or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stoploss upward: raise stop as price moves up
                atr_stop = max(atr_stop, curr_close - 2.5 * atr_val)
        elif position == -1:
            # Short position management
            # Stoploss: price closes above ATR stop level
            if curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            # Exit: price returns above Donchian upper band or trend breaks
            elif curr_close >= donchian_high or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stoploss downward: lower stop as price moves down
                atr_stop = min(atr_stop, curr_close + 2.5 * atr_val)
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0