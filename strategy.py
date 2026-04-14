#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ADX trend filter and volume confirmation
# Uses Donchian channel breakout (20-period) from 4h timeframe for entry signals
# 1d ADX (14) provides trend strength filter to avoid low-momentum breakouts
# Volume confirmation (>1.5x average volume) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of daily trend
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1d ADX (14) for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
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
        
        # Initial values
        tr_sum[period-1] = np.nansum(tr[:period])
        dm_plus_sum[period-1] = np.nansum(dm_plus[:period])
        dm_minus_sum[period-1] = np.nansum(dm_minus[:period])
        
        # Wilder's smoothing
        for i in range(period, len(tr)):
            tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr[i]
            dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / period) + dm_plus[i]
            dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / period) + dm_minus[i]
        
        # DI values
        di_plus = np.where(tr_sum != 0, 100 * dm_plus_sum / tr_sum, 0)
        di_minus = np.where(tr_sum != 0, 100 * dm_minus_sum / tr_sum, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = np.zeros_like(dx)
        if len(dx) >= 2*period-1:
            adx[2*period-2] = np.nansum(dx[period-1:2*period-1]) / period
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for Donchian and ADX calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian high with volume filter and strong trend
            if price > donchian_high[i] and vol > 1.5 * avg_vol[i] and strong_trend:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume filter and strong trend
            elif price < donchian_low[i] and vol > 1.5 * avg_vol[i] and strong_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low (reversal) or ADX weakens
            if price < donchian_low[i] or adx_1d_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high (reversal) or ADX weakens
            if price > donchian_high[i] or adx_1d_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_1dADX_Volume"
timeframe = "4h"
leverage = 1.0