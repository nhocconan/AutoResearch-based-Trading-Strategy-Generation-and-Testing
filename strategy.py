#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1w ADX trend filter
# Long when: price breaks above Donchian(20) high AND 1d volume > 2.0x 20-period MA AND 1w ADX > 25
# Short when: price breaks below Donchian(20) low AND 1d volume > 2.0x 20-period MA AND 1w ADX > 25
# Exit when: price returns inside Donchian(20) channel OR 1w ADX < 20 (trend weakens)
# Uses Donchian for structure, volume for conviction, HTF ADX for regime filter
# Timeframe: 4h, HTF: 1d for volume, 1w for ADX. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_1dVolumeSpike_1wADX_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian(20) on 4h
    lookback = 20
    if len(high) >= lookback:
        # Donchian high: highest high over last 20 periods (including current)
        donch_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
        # Donchian low: lowest low over last 20 periods (including current)
        donch_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    else:
        donch_high = np.full(n, np.nan)
        donch_low = np.full(n, np.nan)
    
    # Breakout signals
    breakout_up = close > donch_high  # price above upper band
    breakout_down = close < donch_low  # price below lower band
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    if len(volume_1d) >= 20:
        vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume_1d > (2.0 * vol_ma_20)  # volume > 2x 20-period MA
    else:
        volume_spike = np.zeros(len(df_1d), dtype=bool)
    
    # Get 1w data ONCE before loop for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # need sufficient data for ADX
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on 1w
    if len(high_1w) >= 14:
        # True Range
        tr1 = np.abs(high_1w[1:] - low_1w[1:])
        tr2 = np.abs(high_1w[1:] - close_1w[:-1])
        tr3 = np.abs(low_1w[1:] - close_1w[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Directional Movement
        up_move = high_1w[1:] - high_1w[:-1]
        down_move = low_1w[:-1] - low_1w[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Wilder's smoothing
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        atr = wilder_smooth(tr, 14)
        plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
        minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, 14)
    else:
        adx = np.full(len(df_1w), np.nan)
    
    # ADX trend filter: ADX > 25 = strong trend
    adx_trend = adx > 25
    adx_weak = adx < 20  # for exit condition
    
    # Align HTF indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    adx_trend_aligned = align_htf_to_ltf(prices, df_1w, adx_trend.astype(float))
    adx_weak_aligned = align_htf_to_ltf(prices, df_1w, adx_weak.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_trend_aligned[i]) or 
            np.isnan(adx_weak_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish breakout + volume spike + strong trend
            if (breakout_up[i] and 
                volume_spike_aligned[i] == 1.0 and 
                adx_trend_aligned[i] == 1.0):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish breakout + volume spike + strong trend
            elif (breakout_down[i] and 
                  volume_spike_aligned[i] == 1.0 and 
                  adx_trend_aligned[i] == 1.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns inside channel OR trend weakens
            if (close[i] >= donch_low[i] and close[i] <= donch_high[i]) or adx_weak_aligned[i] == 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns inside channel OR trend weakens
            if (close[i] >= donch_low[i] and close[i] <= donch_high[i]) or adx_weak_aligned[i] == 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals