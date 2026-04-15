#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for HTF context
    daily = get_htf_data(prices, '1d')
    daily_close = daily['close'].values
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_volume = daily['volume'].values
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, daily, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, daily, donchian_low)
    
    # Calculate daily volume average (20-period)
    vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h = align_htf_to_ltf(prices, daily, vol_ma_20)
    
    # Calculate daily ADX for trend strength (14-period)
    # True Range
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]])) > 
                       (np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low), 
                       np.maximum(daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low) > 
                        (daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]])), 
                        np.maximum(np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low, 0), 0)
    
    # Smoothed values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / (atr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (atr_14 + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, daily, adx)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or 
            np.isnan(vol_ma_12h[i]) or np.isnan(adx_12h[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above daily Donchian high with volume confirmation and weak trend (choppy market)
        if (close[i] > donchian_high_12h[i] and 
            volume[i] > vol_ma_12h[i] * 1.5 and  # Volume spike
            adx_12h[i] < 25):  # Choppy/range market (ADX < 25)
            signals[i] = 0.25
        # Short condition: price breaks below daily Donchian low with volume confirmation and weak trend
        elif (close[i] < donchian_low_12h[i] and 
              volume[i] > vol_ma_12h[i] * 1.5 and  # Volume spike
              adx_12h[i] < 25):  # Choppy/range market (ADX < 25)
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_DailyDonchian_Breakout_Volume_ADXFilter"
timeframe = "12h"
leverage = 1.0