#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Uses 1d EMA50 for trend direction and Donchian channels from 4h for entry/exit
# Volume confirmation requires 2.0x average volume to ensure strong participation
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag on 4h timeframe
# Works in both bull and bear markets by following the 1d trend direction and using Donchian for structure

name = "4h_Donchian20_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels from 4h data (using completed bars only)
    # We'll calculate these on the 4h data itself, then align
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 20-period Donchian channels on 4h data
    # Upper channel: highest high over last 20 periods
    # Lower channel: lowest low over last 20 periods
    high_roll = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (already on 4h, so no alignment needed for the channels themselves)
    # But we need to align the 1d EMA to 4h timeframe
    # The Donchian channels are already calculated on 4h data, so we need to align them to the LTF
    # However, since we're using 4h as primary timeframe, we need to map the 4h Donchian to the 4h bars
    # Actually, we're using 4h as both the primary timeframe and for Donchian calculation
    # So we can use the 4h data directly, but we need to align it to the LTF if we were using a lower TF
    # Since we're using 4h as primary, we can use the 4h data directly
    
    # We need to align the 4h Donchian channels to the 4h timeframe (trivial - they're already aligned)
    # But we need to make sure we're using completed bars only
    # The rolling calculation with min_periods=20 ensures we only get values after 20 periods
    
    # Now we need to align these 4h values to our 4h prices (which are the same)
    # Actually, since we're using 4h as primary timeframe, the prices are already 4h bars
    # So we can use the rolling values directly
    
    # But wait, we need to be careful: the prices DataFrame is already 4h bars
    # So we can calculate Donchian directly on the prices
    
    # Let's recalculate: calculate Donchian on the prices (which are 4h bars)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Donchian breakout with 1d trend filter
        # Long: Price breaks above upper Donchian + volume spike + price above 1d EMA50 (uptrend)
        # Short: Price breaks below lower Donchian + volume spike + price below 1d EMA50 (downtrend)
        if position == 0:
            if (close[i] > high_roll[i] and volume_spike and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < low_roll[i] and volume_spike and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below lower Donchian (reversal) OR price below 1d EMA50 (trend change)
            if close[i] < low_roll[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above upper Donchian (reversal) OR price above 1d EMA50 (trend change)
            if close[i] > high_roll[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals