#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w ADX trend filter
# - Long when price breaks above 12h Donchian upper band AND 1d volume > 1.5x 20-period volume SMA AND 1w ADX > 25
# - Short when price breaks below 12h Donchian lower band AND 1d volume > 1.5x 20-period volume SMA AND 1w ADX > 25
# - Exit: price retreats to 12h Donchian middle band or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 12h timeframe to stay within fee drag limits
# - Uses Donchian channels from 12h timeframe for structure, with volume and trend confirmation from higher timeframes

name = "12h_1d_1w_donchian_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (high_max_20 + low_min_20) / 2.0
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1w ADX for trend filter (ADX > 25 indicates trending market)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Calculate smoothed TR and DM (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_period = 14
    atr_1w = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1w
    di_minus = 100 * dm_minus_smooth / atr_1w
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx[di_plus + di_minus == 0] = 0  # Avoid division by zero
    
    def wilders_smoothing_adx(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    adx_1w = wilders_smoothing_adx(dx, atr_period)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Align 12h Donchian levels to 12h timeframe (no shift needed as primary TF)
    # Align 1d volume confirmation to 12h timeframe
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 12h volume SMA for confirmation
    volume_sma_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(volume_sma_20_12h_aligned[i]) or
            np.isnan(volume_sma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA AND 1d volume > 1.5x 20-period volume SMA
        vol_confirm_12h = volume[i] > 1.5 * volume_sma_20_12h[i]
        vol_confirm_1d = volume_1d[i // 4] > 1.5 * volume_sma_20_12h_aligned[i] if i // 4 < len(volume_1d) else False
        vol_confirm = vol_confirm_12h and vol_confirm_1d
        
        # Trend filter: 1w ADX > 25 (trending market)
        trending = adx_1w_aligned[i] > 25.0
        
        # Donchian breakout signals
        breakout_up = close[i] > high_max_20[i-1]  # Break above previous upper band
        breakout_down = close[i] < low_min_20[i-1]  # Break below previous lower band
        
        # Exit conditions: price retreats to middle band or loss of volume confirmation
        exit_long = close[i] < donchian_middle[i] or not vol_confirm
        exit_short = close[i] > donchian_middle[i] or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trending and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and trending and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals