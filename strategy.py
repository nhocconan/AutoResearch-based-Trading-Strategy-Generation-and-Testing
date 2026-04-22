#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation
# Long when price breaks above Donchian upper + ADX > 20 + volume spike
# Short when price breaks below Donchian lower + ADX > 20 + volume spike
# Exit when price crosses Donchian midpoint or ADX drops below 15
# Designed for low trade frequency (~20-40/year) with strong trend-following edge in both bull and bear markets
# Uses Donchian channels for price channels and ADX for trend strength filtering

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily timeframe
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smooth(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            result[period-1] = np.nansum(x[:period])
            for i in range(period, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian(20) on 12h timeframe
    high = prices['high'].values
    low = prices['low'].values
    donchian_up = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_down = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_up + donchian_down) / 2
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(donchian_up[i]) or 
            np.isnan(donchian_down[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        upper = donchian_up[i]
        lower = donchian_down[i]
        mid = donchian_mid[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        # Trend filter: ADX > 20 indicates strong trend
        strong_trend = adx_val > 20
        weak_trend = adx_val < 15
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + strong trend + volume spike
            if price > upper and strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + strong trend + volume spike
            elif price < lower and strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses midpoint or trend weakens
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below midpoint or trend weakens
                if price < mid or weak_trend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above midpoint or trend weakens
                if price > mid or weak_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dADX14_Volume"
timeframe = "12h"
leverage = 1.0