#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channels with volume confirmation and ADX trend filter.
# Long when price breaks above 4h upper Donchian channel (20-period), ADX > 25 (trending), and volume > 1.5x average.
# Short when price breaks below 4h lower Donchian channel (20-period), ADX > 25, and volume > 1.5x average.
# Exit when price returns to Donchian middle or ADX drops below 20 (trend weakening).
# Uses Donchian channels for volatility-based breakout signals, ADX for trend strength confirmation,
# and volume for institutional participation confirmation. Designed to work in both bull and bear markets
# by only trading in trending conditions (ADX > 25) and avoiding choppy markets.
# Target: 15-37 trades/year per symbol (60-150 total over 4 years) to minimize fee drag.
# Uses 4h for signal direction and 1h only for entry timing to reduce whipsaw.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for Donchian channels and ADX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need enough for Donchian(20) and ADX(14)
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian Channels (20-period)
    upper_dc = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    middle_dc = (upper_dc + lower_dc) / 2
    
    # Calculate ADX (14)
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 1h timeframe
    upper_dc_aligned = align_htf_to_ltf(prices, df_4h, upper_dc)
    lower_dc_aligned = align_htf_to_ltf(prices, df_4h, lower_dc)
    middle_dc_aligned = align_htf_to_ltf(prices, df_4h, middle_dc)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(34, 20)  # Need ADX and Donchian periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_dc_aligned[i]) or 
            np.isnan(lower_dc_aligned[i]) or
            np.isnan(middle_dc_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Weak trend filter: ADX < 20 indicates trend weakening
        weak_trend = adx_aligned[i] < 20
        
        if position == 0:
            # Look for Donchian channel breakouts in strong trend
            # Long: price breaks above upper DC AND strong trend AND volume confirmation
            if (close[i] > upper_dc_aligned[i] and 
                strong_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower DC AND strong trend AND volume confirmation
            elif (close[i] < lower_dc_aligned[i] and 
                  strong_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle DC or trend weakens
            if (close[i] <= middle_dc_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle DC or trend weakens
            if (close[i] >= middle_dc_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_Donchian_Channels_ADX_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0