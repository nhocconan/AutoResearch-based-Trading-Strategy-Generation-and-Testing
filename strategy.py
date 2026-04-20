#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for regime and trend filters
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA(50) for long-term trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily 20-period Donchian channels for volatility regime
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_width_20 = donch_high_20 - donch_low_20
    donch_width_ma_50 = pd.Series(donch_width_20).rolling(window=50, min_periods=50).mean().values
    donch_width_ma_50_aligned = align_htf_to_ltf(prices, df_1d, donch_width_ma_50)
    
    # 4h price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h 20-period Donchian breakout levels
    donch_high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donch_width_ma_50_aligned[i]) or 
            np.isnan(donch_high_4h[i]) or np.isnan(donch_low_4h[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_50_1d_aligned[i]
        donch_high = donch_high_4h[i]
        donch_low = donch_low_4h[i]
        donch_width_ma = donch_width_ma_50_aligned[i]
        vol_ratio_4h = vol_ratio[i]
        
        # Trend filter: price above/below daily EMA50
        trend_up = price > ema_trend
        trend_down = price < ema_trend
        
        # Volatility regime filter: avoid low volatility (chop)
        vol_filter = donch_width_20[i] > 0.5 * donch_width_ma
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_4h > 1.5)
        
        if position == 0:
            # Enter long on Donchian breakout with trend and volume
            if price > donch_high and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short on Donchian breakdown with trend and volume
            elif price < donch_low and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retrace to midpoint or breakdown
            midpoint = (donch_high + donch_low) / 2
            if price < midpoint or price < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retrace to midpoint or breakout
            midpoint = (donch_high + donch_low) / 2
            if price > midpoint or price > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_DonchianBreakout_EMA50_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0