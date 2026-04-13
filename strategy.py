#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly ADX filter and volume confirmation.
# Daily timeframe reduces trade frequency to avoid fee drag.
# Weekly ADX ensures we only trade in trending markets (ADX > 25).
# Volume confirmation ensures breakouts have conviction.
# Works in both bull and bear markets by capturing strong trends.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on daily data
    period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(period-1, n):
        donchian_high[i] = np.max(high[i-period+1:i+1])
        donchian_low[i] = np.min(low[i-period+1:i+1])
    
    # Calculate ATR (14-period) for stop loss
    atr_period = 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate weekly ADX (14-period) for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.zeros(len(close_1w))
    for i in range(1, len(close_1w)):
        tr1[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(len(close_1w))
    minus_dm = np.zeros(len(close_1w))
    for i in range(1, len(close_1w)):
        up = high_1w[i] - high_1w[i-1]
        down = low_1w[i-1] - low_1w[i]
        if up > down and up > 0:
            plus_dm[i] = up
        else:
            plus_dm[i] = 0
        if down > up and down > 0:
            minus_dm[i] = down
        else:
            minus_dm[i] = 0
    
    # Smoothed values
    atr_1w = np.zeros(len(close_1w))
    for i in range(atr_period, len(close_1w)):
        atr_1w[i] = np.mean(tr1[i-atr_period+1:i+1])
    
    plus_dm_smooth = np.zeros(len(close_1w))
    minus_dm_smooth = np.zeros(len(close_1w))
    for i in range(atr_period, len(close_1w)):
        plus_dm_smooth[i] = np.mean(plus_dm[i-atr_period+1:i+1])
        minus_dm_smooth[i] = np.mean(minus_dm[i-atr_period+1:i+1])
    
    # DI and DX
    plus_di = np.zeros(len(close_1w))
    minus_di = np.zeros(len(close_1w))
    dx = np.zeros(len(close_1w))
    for i in range(atr_period, len(close_1w)):
        if atr_1w[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_1w[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_1w[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX
    adx_1w = np.zeros(len(close_1w))
    for i in range(2*atr_period-1, len(close_1w)):
        adx_1w[i] = np.mean(dx[i-atr_period+1:i+1])
    
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        atr_val = atr[i]
        adx_val = adx_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Trend filter: weekly ADX > 25 indicates strong trend
        trend_filter = adx_val > 25
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + strong trend
            if (price > donch_high and 
                volume_confirm and 
                trend_filter):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low + volume + strong trend
            elif (price < donch_low and 
                  volume_confirm and 
                  trend_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low OR trend weakens
            if (price < donch_low or 
                adx_val < 20):  # Exit when trend weakens
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR trend weakens
            if (price > donch_high or 
                adx_val < 20):  # Exit when trend weakens
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Breakout_ADX_Filter_v1"
timeframe = "1d"
leverage = 1.0