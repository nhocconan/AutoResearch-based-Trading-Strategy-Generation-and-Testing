#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with volume confirmation and ADX filter
# - Long: Price breaks above 1d Donchian high with volume > 1.5x 20-period average and ADX > 20
# - Short: Price breaks below 1d Donchian low with volume > 1.5x 20-period average and ADX > 20
# - Exit: Opposite Donchian level touch or ADX < 15 (trend weakening)
# - Uses 1d timeframe for structural breaks to reduce frequency and avoid noise
# - Volume and ADX filters ensure trades only in strong momentum conditions
# - Target: 20-50 total trades over 4 years (5-12/year) with 0.30 position sizing

name = "4h_1dDonchian20_Volume_ADX_Breakout"
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
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period high/low)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # ADX filter (14-period)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    up_move = np.concatenate([[np.nan], high[1:] - high[:-1]])
    down_move = np.concatenate([[np.nan], low[:-1] - low[1:]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR and DM using Wilder's smoothing (EMA with alpha=1/period)
    atr = np.full(n, np.nan)
    plus_dm_smooth = np.full(n, np.nan)
    minus_dm_smooth = np.full(n, np.nan)
    
    # Wilder smoothing: first value is average of first 'period' values
    if n >= 14:
        atr[13] = np.nanmean(tr[1:15])  # Average of first 14 TR values (indices 1-14)
        plus_dm_smooth[13] = np.nanmean(plus_dm[1:15])
        minus_dm_smooth[13] = np.nanmean(minus_dm[1:15])
        
        # Subsequent values: smoothed = previous_smoothed - (previous_smoothed/period) + current_value
        for i in range(14, n):
            atr[i] = atr[i-1] - (atr[i-1]/14) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1]/14) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1]/14) + minus_dm[i]
    
    # Calculate DI values
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    
    for i in range(14, n):
        if not np.isnan(atr[i]) and atr[i] != 0:
            plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # Calculate ADX (smoothed DX)
    adx = np.full(n, np.nan)
    if n >= 27:  # Need 14 for DX + 14 more for smoothing
        # First ADX value is average of first 14 DX values
        dx_sum = 0
        count = 0
        for i in range(14, 28):
            if not np.isnan(dx[i]):
                dx_sum += dx[i]
                count += 1
        if count > 0:
            adx[27] = dx_sum / count
        
        # Subsequent ADX values: smoothed = previous_adx - (previous_adx/14) + current_dx
        for i in range(28, n):
            if not np.isnan(dx[i]) and not np.isnan(adx[i-1]):
                adx[i] = adx[i-1] - (adx[i-1]/14) + dx[i]
    
    # ADX filter: trend strength > 20
    adx_filter = adx > 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or
            np.isnan(volume_filter[i]) or np.isnan(adx_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above 1d Donchian high with volume and ADX confirmation
            if close[i] > donchian_high_4h[i] and volume_filter[i] and adx_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below 1d Donchian low with volume and ADX confirmation
            elif close[i] < donchian_low_4h[i] and volume_filter[i] and adx_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price touches 1d Donchian low or ADX weakens
            if close[i] <= donchian_low_4h[i] or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price touches 1d Donchian high or ADX weakens
            if close[i] >= donchian_high_4h[i] or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals