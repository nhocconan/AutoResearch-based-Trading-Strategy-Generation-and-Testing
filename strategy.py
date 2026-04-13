#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ADX(14) trend filter and volume confirmation.
# Donchian breakouts capture momentum; ADX ensures trend strength (>25) to avoid false breakouts in ranging markets.
# Volume surge confirms breakout validity. Target: 15-25 trades/year (60-100 total) for 1d timeframe.
# Works in bull (breakouts up) and bear (breakouts down) via symmetric long/short logic.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on 1w
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        atr = np.full_like(tr, np.nan)
        plus_dm_smooth = np.full_like(plus_dm, np.nan)
        minus_dm_smooth = np.full_like(minus_dm, np.nan)
        
        # First values: simple average
        if len(tr) >= period + 1:
            atr[period] = np.nanmean(tr[1:period+1])
            plus_dm_smooth[period] = np.nanmean(plus_dm[1:period+1])
            minus_dm_smooth[period] = np.nanmean(minus_dm[1:period+1])
            
            # Subsequent values: Wilder smoothing
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full_like(tr, np.nan)
        minus_di = np.full_like(tr, np.nan)
        dx = np.full_like(tr, np.nan)
        
        for i in range(period, len(tr)):
            if atr[i] != 0 and not np.isnan(atr[i]):
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX: smoothed DX
        adx = np.full_like(tr, np.nan)
        if len(tr) >= 2*period + 1:
            adx[2*period] = np.nanmean(dx[period+1:2*period+1])
            for i in range(2*period+1, len(tr)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Donchian(20) channels on 1d
    def calculate_donchian(high, low, window):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(window-1, len(high)):
            upper[i] = np.max(high[i-window+1:i+1])
            lower[i] = np.min(low[i-window+1:i+1])
        return upper, lower
    
    upper_channel, lower_channel = calculate_donchian(high, low, 20)
    
    # Average volume (20-period) for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        adx_val = adx_1w_aligned[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: breakout above upper channel + ADX > 25 + volume confirmation
            if (price > upper_channel[i] and adx_val > 25 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: breakout below lower channel + ADX > 25 + volume confirmation
            elif (price < lower_channel[i] and adx_val > 25 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price re-enters Donchian channel (below midpoint) or ADX weakens
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if (price < midpoint or adx_val < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price re-enters Donchian channel (above midpoint) or ADX weakens
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if (price > midpoint or adx_val < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_ADX_Volume_Breakout"
timeframe = "1d"
leverage = 1.0