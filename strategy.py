#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ADX trend filter
# Long when price breaks above Donchian upper channel with volume spike and ADX > 25
# Short when price breaks below Donchian lower channel with volume spike and ADX > 25
# Uses daily ADX for trend strength filter to avoid whipsaws in ranging markets
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years)

name = "4h_DonchianBreakout_Volume_ADX"
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
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX components on daily data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_period = 14
    atr = np.zeros_like(tr)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    # Initial values
    atr[tr_period] = np.mean(tr[:tr_period+1])
    plus_dm_smooth[tr_period] = np.mean(plus_dm[:tr_period+1])
    minus_dm_smooth[tr_period] = np.mean(minus_dm[:tr_period+1])
    
    # Wilder's smoothing
    for i in range(tr_period+1, len(tr)):
        atr[i] = (atr[i-1] * (tr_period-1) + tr[i]) / tr_period
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (tr_period-1) + plus_dm[i]) / tr_period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (tr_period-1) + minus_dm[i]) / tr_period
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # ADX calculation
    adx_period = 14
    adx = np.zeros_like(dx)
    adx[2*adx_period] = np.mean(dx[adx_period:2*adx_period+1])
    for i in range(2*adx_period+1, len(dx)):
        adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align daily ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 2*adx_period+1)  # Need Donchian and ADX data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_channel = high_max[i]
        lower_channel = low_min[i]
        adx_val = adx_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        # ADX trend filter: only trade when trend is strong enough
        strong_trend = adx_val > 25
        
        if position == 0:
            # Enter long: price breaks above upper channel with volume and trend
            if price > upper_channel and volume_confirmed and strong_trend:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel with volume and trend
            elif price < lower_channel and volume_confirmed and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below lower channel or trend weakens
            if price < lower_channel or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above upper channel or trend weakens
            if price > upper_channel or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals