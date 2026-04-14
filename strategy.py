#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d ADX Trend Filter and Volume Confirmation
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13)
# Long when Bull Power > 0 and Bear Power < 0 with ADX > 25 (trending)
# Short when Bear Power < 0 and Bull Power > 0 with ADX > 25
# Volume > 1.5x average confirms institutional participation
# Designed to capture strong trends in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX and EMA13
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 13-period EMA for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    atr[0:13] = np.nan
    
    di_plus = 100 * pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values / atr
    di_minus = 100 * pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 1d indicators to 6h
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Elder Ray components on 6h data
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13_6h
    bear_power = low - ema13_6h
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA and ADX calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Bull Power > 0 and Bear Power < 0 with strong trend and volume
            if bull_power[i] > 0 and bear_power[i] < 0 and strong_trend and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: Bear Power < 0 and Bull Power > 0 with strong trend and volume
            elif bear_power[i] < 0 and bull_power[i] > 0 and strong_trend and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power <= 0 or Bear Power >= 0 (trend weakening)
            if bull_power[i] <= 0 or bear_power[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bull Power >= 0 or Bear Power <= 0 (trend weakening)
            if bull_power[i] >= 0 or bear_power[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_1dADX_Volume"
timeframe = "6h"
leverage = 1.0