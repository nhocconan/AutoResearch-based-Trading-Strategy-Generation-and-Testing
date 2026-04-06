#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ADX trend filter
# Long when price breaks above Donchian(20) high with volume > 1.5x average and ADX > 25
# Short when price breaks below Donchian(20) low with volume > 1.5x average and ADX > 25
# Exit when price crosses Donchian midline or volume drops below average
# Uses 4h timeframe with volume and trend confirmation to reduce false breakouts
# Targets 100-200 total trades over 4 years (25-50/year) with focus on strong trending moves

name = "4h_donchian_vol_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # ADX (14-period) for trend strength
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus_sum = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus_sum = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_sum / (tr_sum + 1e-10)
    di_minus = 100 * dm_minus_sum / (tr_sum + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx = adx.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(adx[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses Donchian midline OR weak trend
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume and trend confirmation
            # Bullish breakout: price above Donchian high with volume and ADX
            if (close[i] > donchian_high[i] and 
                volume[i] > volume_threshold[i] and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price below Donchian low with volume and ADX
            elif (close[i] < donchian_low[i] and 
                  volume[i] > volume_threshold[i] and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
    
    return signals