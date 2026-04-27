#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation.
# Long when price breaks above 20-period high with 1d ADX > 25 and volume > 1.5x average.
# Short when price breaks below 20-period low with 1d ADX > 25 and volume > 1.5x average.
# Exit when price crosses 10-period SMA in opposite direction.
# Uses Donchian for breakout, ADX for trend strength, volume for confirmation.
# Targets 15-30 trades/year to avoid fee drag while capturing strong trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(high[1:] - low[1:], 
                        np.maximum(np.abs(high[1:] - close[:-1]), 
                                   np.abs(low[1:] - close[:-1])))
        tr = np.concatenate([[high[0] - low[0]], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr = np.zeros(n)
        dm_plus_smooth = np.zeros(n)
        dm_minus_smooth = np.zeros(n)
        
        atr[period-1] = np.mean(tr[:period])
        dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
        dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
        
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # DI and DX
        di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
        di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
        dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        
        # ADX
        adx = np.full(n, np.nan)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate Donchian channels (20-period)
    donchian_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper_channel[i] = np.max(high[i - donchian_period + 1:i + 1])
        lower_channel[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Calculate 10-period SMA for exit
    sma_period = 10
    sma = np.full(n, np.nan)
    for i in range(sma_period - 1, n):
        sma[i] = np.mean(close[i - sma_period + 1:i + 1])
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, ADX, SMA, and volume MA20
    start_idx = max(donchian_period - 1, 2*14-1, sma_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(sma[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter: require ADX > 25
        trend_filter = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above upper Donchian with trend and volume
            if price > upper_channel[i] and trend_filter and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian with trend and volume
            elif price < lower_channel[i] and trend_filter and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 10-period SMA
            if price < sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 10-period SMA
            if price > sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_Breakout_1dADX14_Volume"
timeframe = "12h"
leverage = 1.0