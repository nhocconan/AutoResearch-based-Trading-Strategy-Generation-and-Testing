#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d ADX Trend Filter and Volume Confirmation
# Long when Bull Power > 0, Bear Power < 0, ADX > 20 (trending), and 1d volume > 1.5x average
# Short when Bear Power < 0, Bull Power > 0, ADX > 20, and 1d volume > 1.5x average
# Exit when trend weakens (ADX < 20) or volume drops below average
# Designed to capture institutional buying/selling pressure in trending markets with volume confirmation
# Works in both bull and bear markets by following the trend direction with Elder Ray confirming momentum
# Target: 60-120 total trades over 4 years (15-30/year)

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 13-period EMA for Elder Ray (using 1d close)
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # Calculate 1d ADX for trend strength
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for EMA and ADX calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: Bull Power > 0 (buying pressure), Bear Power < 0, strong trend, volume confirmation
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                adx_aligned[i] > 20 and 
                df_1d['volume'].values[i] if i < len(df_1d['volume'].values) else df_1d['volume'].values[-1] > 1.5 * vol_ma_20_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: Bear Power < 0 (selling pressure), Bull Power > 0, strong trend, volume confirmation
            elif (bear_power_aligned[i] < 0 and 
                  bull_power_aligned[i] > 0 and 
                  adx_aligned[i] > 20 and 
                  df_1d['volume'].values[i] if i < len(df_1d['volume'].values) else df_1d['volume'].values[-1] > 1.5 * vol_ma_20_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend weakening (ADX < 20) or volume dropping
            if adx_aligned[i] < 20 or df_1d['volume'].values[i] if i < len(df_1d['volume'].values) else df_1d['volume'].values[-1] < vol_ma_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend weakening (ADX < 20) or volume dropping
            if adx_aligned[i] < 20 or df_1d['volume'].values[i] if i < len(df_1d['volume'].values) else df_1d['volume'].values[-1] < vol_ma_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_ADX_Volume"
timeframe = "6h"
leverage = 1.0