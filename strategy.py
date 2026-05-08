#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h ADX trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high in 12h uptrend (ADX>25 and +DI>-DI).
# Short when price breaks below Donchian(20) low in 12h downtrend (ADX>25 and +DI< -DI).
# Volume > 1.5x 20-period average confirms participation. Works in both bull and bear markets
# by taking breakouts in the direction of the higher timeframe trend.
# Target: 25-40 trades/year with disciplined entries to minimize fee drag.

name = "4h_Donchian_Breakout_12hADX_Volume"
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
    
    # Get 12h data for ADX trend filter and volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate ADX(14) on 12h data
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period * 2:
            return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0  # first value
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr = np.full(n, np.nan)
        dm_plus_smooth = np.full(n, np.nan)
        dm_minus_smooth = np.full(n, np.nan)
        
        if n >= period:
            atr[period-1] = np.mean(tr[:period])
            dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
            dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
            
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # DI and DX
        di_plus = np.full(n, np.nan)
        di_minus = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        for i in range(period, n):
            if atr[i] != 0:
                di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
                di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
                if di_plus[i] + di_minus[i] != 0:
                    dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        
        # ADX
        adx = np.full(n, np.nan)
        if n >= 2*period:
            adx[2*period-1] = np.mean(dx[period:2*period])
            for i in range(2*period, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx, di_plus, di_minus
    
    adx_12h, di_plus_12h, di_minus_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Calculate 20-period average volume on 12h
    vol_avg_20_12h = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        vol_avg_20_12h[i] = np.mean(volume_12h[i-20:i])
    
    # Calculate Donchian channels (20-period) on 4h data
    def calculate_donchian(high, low, period=20):
        n = len(high)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        for i in range(period-1, n):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Align all indicators to 4h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    di_plus_12h_aligned = align_htf_to_ltf(prices, df_12h, di_plus_12h)
    di_minus_12h_aligned = align_htf_to_ltf(prices, df_12h, di_minus_12h)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(di_plus_12h_aligned[i]) or 
            np.isnan(di_minus_12h_aligned[i]) or np.isnan(vol_avg_20_12h_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 12h volume > 1.5x 20-period average
        vol_filter = False
        if not np.isnan(vol_avg_20_12h_aligned[i]):
            # Find current 12h bar's volume
            idx_12h = 0
            while idx_12h < len(df_12h) and df_12h.iloc[idx_12h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_12h += 1
            idx_12h -= 1  # last completed 12h bar
            
            if idx_12h >= 0:
                vol_12h_current = df_12h.iloc[idx_12h]['volume']
                vol_filter = vol_12h_current > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Trend filter: ADX > 25 and directional bias
        uptrend = (adx_12h_aligned[i] > 25) and (di_plus_12h_aligned[i] > di_minus_12h_aligned[i])
        downtrend = (adx_12h_aligned[i] > 25) and (di_plus_12h_aligned[i] < di_minus_12h_aligned[i])
        
        if position == 0:
            # Look for entry: Donchian breakout + trend + volume
            long_condition = (close[i] > donchian_upper_aligned[i]) and uptrend and vol_filter
            short_condition = (close[i] < donchian_lower_aligned[i]) and downtrend and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Donchian lower or trend changes
            if (close[i] < donchian_lower_aligned[i]) or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Donchian upper or trend changes
            if (close[i] > donchian_upper_aligned[i]) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals