#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian channel breakout with 1d ADX trend filter and volume confirmation
    # Donchian channels capture breakouts from volatility contractions
    # ADX > 25 on daily confirms trending regime (avoids whipsaws in ranges)
    # Volume spike (1.5x 20-period MA) confirms institutional participation
    # Works in both bull and bear markets by following established trends
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high - np.concatenate([[high[0]], high[:-1]])) > 
                          (np.concatenate([[low[0]], low[:-1]]) - low), 
                          np.maximum(high - np.concatenate([[high[0]], high[:-1]]), 0), 0)
        dm_minus = np.where((np.concatenate([[low[0]], low[:-1]]) - low) > 
                           (high - np.concatenate([[high[0]], high[:-1]])), 
                           np.maximum(np.concatenate([[low[0]], low[:-1]]) - low, 0), 0)
        
        # Smoothed values
        tr_sum = np.zeros_like(tr)
        dm_plus_sum = np.zeros_like(dm_plus)
        dm_minus_sum = np.zeros_like(dm_minus)
        
        # Initial smoothed values (first period)
        tr_sum[period-1] = np.nansum(tr[:period])
        dm_plus_sum[period-1] = np.nansum(dm_plus[:period])
        dm_minus_sum[period-1] = np.nansum(dm_minus[:period])
        
        # Wilder's smoothing
        for i in range(period, len(tr)):
            tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr[i]
            dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / period) + dm_plus[i]
            dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / period) + dm_minus[i]
        
        # Directional Indicators
        di_plus = 100 * dm_plus_sum / tr_sum
        di_minus = 100 * dm_minus_sum / tr_sum
        
        # DX and ADX
        dx = np.zeros_like(tr)
        mask = (di_plus + di_minus) != 0
        dx[mask] = 100 * np.abs(di_plus[mask] - di_minus[mask]) / (di_plus[mask] + di_minus[mask])
        
        adx = np.zeros_like(tr)
        adx[2*period-2:] = np.nan
        if len(tr) >= 2*period-1:
            adx[2*period-1] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period, len(tr)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Donchian Channels (20-period) on 4h
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    dc_upper, dc_lower = donchian_channels(high, low, 20)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper band + volume spike + ADX > 25 (strong trend)
            if close[i] > dc_upper[i] and vol_spike[i] and adx_14_1d_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band + volume spike + ADX > 25 (strong trend)
            elif close[i] < dc_lower[i] and vol_spike[i] and adx_14_1d_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Donchian band or trend weakening (ADX < 20)
            if position == 1:
                if close[i] < dc_lower[i] or adx_14_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > dc_upper[i] or adx_14_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dADX25_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0