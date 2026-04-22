#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h breakout of 12h Donchian Channels with volume confirmation and ADX trend filter
    # Donchian channels represent key support/resistance levels. Breakouts with volume confirm
    # institutional participation. ADX ensures alignment with trending markets.
    # This combination reduces false breakouts and improves win rate in both bull and bear markets.
    # Focus on 4h timeframe with strict entry conditions to limit trades to 20-50/year.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian Channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian Channels (20-period)
    dc_period = 20
    upper_dc = pd.Series(high_12h).rolling(window=dc_period, min_periods=dc_period).max().values
    lower_dc = pd.Series(low_12h).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # Align Donchian Channels to 4h timeframe
    upper_dc_aligned = align_htf_to_ltf(prices, df_12h, upper_dc)
    lower_dc_aligned = align_htf_to_ltf(prices, df_12h, lower_dc)
    
    # Load 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX (14-period)
    adx_period = 14
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr_ma = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).sum().values
    dm_plus_ma = pd.Series(dm_plus).rolling(window=adx_period, min_periods=adx_period).sum().values
    dm_minus_ma = pd.Series(dm_minus).rolling(window=adx_period, min_periods=adx_period).sum().values
    
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    # Handle division by zero
    adx = np.where((di_plus + di_minus) == 0, 0, adx)
    
    # Align ADX to 4h timeframe (already on 4h)
    adx_aligned = adx  # Already on 4h timeframe
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma20  # Require 1.8x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(upper_dc_aligned[i]) or np.isnan(lower_dc_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above upper Donchian with volume + ADX > 25 (trending)
            if close[i] > upper_dc_aligned[i] and vol_spike[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower Donchian with volume + ADX > 25 (trending)
            elif close[i] < lower_dc_aligned[i] and vol_spike[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level or ADX < 20 (range)
            if position == 1:
                if close[i] < lower_dc_aligned[i] or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_dc_aligned[i] or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hDC_4hADX25_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0