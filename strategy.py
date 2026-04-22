#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index with Donchian breakout and volume confirmation
# Uses Choppiness Index to distinguish ranging vs trending markets
# In trending markets (CHOP < 38.2): Donchian breakout with volume confirmation
# In ranging markets (CHOP > 61.8): Mean reversion at Donchian channels
# Designed for low trade frequency with regime adaptation to work in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Choppiness Index calculation
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate True Range and ADX components for Choppiness Index (14-period)
    tr1 = high_daily[1:] - low_daily[1:]
    tr2 = np.abs(high_daily[1:] - close_daily[:-1])
    tr3 = np.abs(low_daily[1:] - close_daily[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_daily[1:] - high_daily[:-1]) > (low_daily[:-1] - low_daily[1:]), 
                       np.maximum(high_daily[1:] - high_daily[:-1], 0), 0)
    dm_minus = np.where((low_daily[:-1] - low_daily[1:]) > (high_daily[1:] - high_daily[:-1]), 
                        np.maximum(low_daily[:-1] - low_daily[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (14-period)
    atr = np.full_like(tr, np.nan, dtype=float)
    dm_plus_smooth = np.full_like(dm_plus, np.nan, dtype=float)
    dm_minus_smooth = np.full_like(dm_minus, np.nan, dtype=float)
    
    # Wilder smoothing: first value is average of first 14 periods
    if len(tr) >= 14:
        atr[13] = np.nansum(tr[1:15])
        dm_plus_smooth[13] = np.nansum(dm_plus[1:15])
        dm_minus_smooth[13] = np.nansum(dm_minus[1:15])
        
        # Subsequent values: smoothed = previous_smoothed - (previous_smoothed/14) + current_value
        for i in range(14, len(tr)):
            atr[i] = atr[i-1] - (atr[i-1]/14) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1]/14) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1]/14) + dm_minus[i]
    
    # Directional Indicators
    di_plus = np.full_like(tr, np.nan, dtype=float)
    di_minus = np.full_like(tr, np.nan, dtype=float)
    dx = np.full_like(tr, np.nan, dtype=float)
    
    for i in range(14, len(tr)):
        if atr[i] != 0:
            di_plus[i] = 100 * (dm_plus_smooth[i] / atr[i])
            di_minus[i] = 100 * (dm_minus_smooth[i] / atr[i])
            if (di_plus[i] + di_minus[i]) != 0:
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX (smoothed DX)
    adx = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= 27:  # Need 14 for DX + 13 more for smoothing
        adx[26] = np.nansum(dx[14:28]) / 14
        for i in range(27, len(tr)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Choppiness Index: CHOP = 100 * log10(SUM(ATR,14) / (MAX(HIGH,14) - MIN(LOW,14))) / log10(14)
    chop = np.full_like(close_daily, np.nan, dtype=float)
    for i in range(13, len(tr)):
        if atr[i] != 0 and not np.isnan(atr[i]):
            sum_atr = np.nansum(tr[max(1, i-12):i+1])  # Sum of ATR over 14 periods
            max_high = np.nanmax(high_daily[max(0, i-13):i+1])
            min_low = np.nanmin(low_daily[max(0, i-13):i+1])
            if max_high > min_low and sum_atr > 0:
                chop[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_daily, chop)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Trending market (CHOP < 38.2): Donchian breakout with volume confirmation
            if chop_val < 38.2:
                # Long breakout
                if price > upper_band and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown
                elif price < lower_band and vol_spike:
                    signals[i] = -0.25
                    position = -1
            # Ranging market (CHOP > 61.8): Mean reversion at Donchian channels
            elif chop_val > 61.8:
                # Long at lower band (oversold)
                if price < lower_band and vol_spike:
                    signals[i] = 0.20
                    position = 1
                # Short at upper band (overbought)
                elif price > upper_band and vol_spike:
                    signals[i] = -0.20
                    position = -1
        
        elif position != 0:
            # Exit conditions based on market regime
            exit_signal = False
            
            if chop_val < 38.2:  # Trending market
                if position == 1:  # long position
                    # Exit when price returns to middle of channel or breaks lower band
                    mid_point = (upper_band + lower_band) / 2
                    if price < mid_point or price < lower_band:
                        exit_signal = True
                elif position == -1:  # short position
                    # Exit when price returns to middle of channel or breaks upper band
                    mid_point = (upper_band + lower_band) / 2
                    if price > mid_point or price > upper_band:
                        exit_signal = True
            else:  # Ranging or transitional market
                # Exit when price returns to opposite band or volatility expands
                if position == 1:  # long position
                    if price > upper_band * 0.95:  # Near upper band
                        exit_signal = True
                elif position == -1:  # short position
                    if price < lower_band * 1.05:  # Near lower band
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.20
    
    return signals

name = "4h_Chop_Donchian_Breakout_MR_Volume"
timeframe = "4h"
leverage = 1.0