#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + weekly ADX trend filter + volume confirmation.
# Long: close > Donchian high (20) + weekly ADX > 25 + volume > 1.5x average volume.
# Short: close < Donchian low (20) + weekly ADX > 25 + volume > 1.5x average volume.
# Works in both bull and bear by using weekly ADX to filter only trending markets.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period)
    donch_high = np.full(len(high_1d), np.nan)
    donch_low = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        donch_high[i] = np.max(high_1d[i-20:i])
        donch_low[i] = np.min(low_1d[i-20:i])
    
    # Weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        plus_dm_smooth = np.full_like(plus_dm, np.nan)
        minus_dm_smooth = np.full_like(minus_dm, np.nan)
        
        if len(tr) >= period:
            # Initial values
            atr[period] = np.nanmean(tr[1:period+1])
            plus_dm_smooth[period] = np.nanmean(plus_dm[1:period+1])
            minus_dm_smooth[period] = np.nanmean(minus_dm[1:period+1])
            
            # Wilder's smoothing
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full_like(plus_dm_smooth, np.nan)
        minus_di = np.full_like(minus_dm_smooth, np.nan)
        dx = np.full_like(tr, np.nan)
        
        mask = ~np.isnan(atr) & (atr != 0)
        plus_di[mask] = 100 * plus_dm_smooth[mask] / atr[mask]
        minus_di[mask] = 100 * minus_dm_smooth[mask] / atr[mask]
        
        dx_mask = ~np.isnan(plus_di) & ~np.isnan(minus_di) & ((plus_di + minus_di) != 0)
        dx[dx_mask] = 100 * np.abs(plus_di[dx_mask] - minus_di[dx_mask]) / (plus_di[dx_mask] + minus_di[dx_mask])
        
        # ADX (smoothed DX)
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            valid_dx = dx[~np.isnan(dx)]
            if len(valid_dx) >= period:
                adx[period-1] = np.nanmean(valid_dx[:period])
                for i in range(period, len(dx)):
                    if not np.isnan(dx[i]):
                        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Average volume (10-period = ~2 weeks) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(10, n):
        avg_volume[i] = np.mean(volume[i-10:i])
    
    # Align indicators to 1d timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_val > 25
        
        if position == 0:
            # Long: break above Donchian high + trending + volume confirmation
            if (price > upper and 
                trending and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below Donchian low + trending + volume confirmation
            elif (price < lower and 
                  trending and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: break below Donchian low or loss of trend
            if (price < lower or
                adx_val < 20):  # Exit when ADX drops below 20 (range)
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: break above Donchian high or loss of trend
            if (price > upper or
                adx_val < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_ADX_Volume"
timeframe = "1d"
leverage = 1.0