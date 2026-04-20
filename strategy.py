#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and ADX trend filter
# - Entry on Donchian(20) breakout in direction of 1d trend (ADX>25)
# - Volume confirmation: current volume > 1.5x 20-period average volume
# - Only long when 1d ADX>25 and +DI>-DI, short when 1d ADX>25 and -DI>+DI
# - Exit when price crosses opposite Donchian band or ADX<20 (trend weakens)
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate ADX on 1d timeframe
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First value has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        for i in range(len(data)):
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = data[i]
            else:
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    plus_di = 100 * wilders_smoothing(dm_plus, period) / atr
    minus_di = 100 * wilders_smoothing(dm_minus, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Align ADX and DI to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_4h = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_4h = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Calculate Donchian channels on 4h
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    vol_4h = prices['volume'].values
    
    donchian_period = 20
    upper_channel = pd.Series(high_4h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low_4h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    volume_confirm = vol_4h > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(adx_4h[i]) or np.isnan(plus_di_4h[i]) or np.isnan(minus_di_4h[i]) or \
           np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(volume_confirm[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend direction from 1d ADX
        strong_trend = adx_4h[i] > 25
        bullish_trend = strong_trend and (plus_di_4h[i] > minus_di_4h[i])
        bearish_trend = strong_trend and (minus_di_4h[i] > plus_di_4h[i])
        weak_trend = adx_4h[i] < 20
        
        if position == 0:
            # Long entry: Donchian breakout up + bullish trend + volume confirmation
            if close_4h[i] > upper_channel[i] and bullish_trend and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Donchian breakout down + bearish trend + volume confirmation
            elif close_4h[i] < lower_channel[i] and bearish_trend and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Donchian breakout down OR trend weakens
            if close_4h[i] < lower_channel[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Donchian breakout up OR trend weakens
            if close_4h[i] > upper_channel[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_1dADX_VolumeFilter"
timeframe = "4h"
leverage = 1.0