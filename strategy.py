#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. ADX filters for trending vs ranging markets.
# In trending markets (ADX > 25), we take Williams %R reversals from extreme levels.
# In ranging markets (ADX < 20), we mean revert at Williams %R overbought/oversold levels.
# Volume confirmation ensures momentum behind moves. Works in both bull and bear markets by
# adapting to market regime via ADX filter. Target: 15-30 trades/year per symbol.

name = "6h_WilliamsR_ADX_Volume_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    # True Range
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(arr[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # Directional Indicators
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Calculate Williams %R(14) on 1d
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) != 0, williams_r, -50)  # Avoid division by zero
    
    # Align indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need Williams %R and ADX data (14+14+7 for smoothing)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        williams_r_val = williams_r_aligned[i]
        
        # Volume and regime filters
        volume_confirmed = vol > 1.3 * vol_ma
        trending_market = adx_val > 25
        ranging_market = adx_val < 20
        
        if position == 0:
            # Long conditions:
            # Trending market: Williams %R crosses above -80 from oversold
            # Ranging market: Williams %R below -80 (oversold) with volume
            if ((trending_market and williams_r_val > -80 and 
                 williams_r_aligned[i-1] <= -80 and volume_confirmed) or
                (ranging_market and williams_r_val < -80 and volume_confirmed)):
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # Trending market: Williams %R crosses below -20 from overbought
            # Ranging market: Williams %R above -20 (overbought) with volume
            elif ((trending_market and williams_r_val < -20 and 
                   williams_r_aligned[i-1] >= -20 and volume_confirmed) or
                  (ranging_market and williams_r_val > -20 and volume_confirmed)):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses below -50 or overbought in ranging market
            if williams_r_val < -50 or (ranging_market and williams_r_val > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses above -50 or oversold in ranging market
            if williams_r_val > -50 or (ranging_market and williams_r_val < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals