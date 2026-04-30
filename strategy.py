#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
# Long when price breaks above 20-period Donchian high, 1d ATR ratio < 0.8 (low volatility regime), and volume > 1.5x 20-bar avg.
# Short when price breaks below 20-period Donchian low, 1d ATR ratio < 0.8, and volume > 1.5x 20-bar avg.
# Exit when price crosses the 10-period Donchian midpoint (mean reversion in low vol regime).
# Uses 4h timeframe for optimal trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Donchian channels provide clear breakout levels with defined risk.
# 1d ATR ratio (ATR(10)/ATR(30)) filters for low volatility regimes where breakouts are more reliable.
# Volume confirmation reduces false breakouts.
# Works in bull markets via breakouts with trend and in bear markets via breakdowns with trend.
# Target: 75-200 total trades over 4 years.

name = "4h_Donchian20_1dATRratio_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d ATR(10) and ATR(30) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_30_1d = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    atr_ratio_1d = atr_10_1d / (atr_30_1d + 1e-10)  # avoid division by zero
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid_10 = (donchian_high_20 + donchian_low_20) / 2  # 10-period midpoint for exit
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_mid_10[i]) or np.isnan(atr_ratio_1d_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high_20[i]
        curr_donchian_low = donchian_low_20[i]
        curr_donchian_mid = donchian_mid_10[i]
        curr_atr_ratio = atr_ratio_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, low volatility regime, volume spike
            if (curr_close > curr_donchian_high and 
                curr_atr_ratio < 0.8 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, low volatility regime, volume spike
            elif (curr_close < curr_donchian_low and 
                  curr_atr_ratio < 0.8 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below Donchian midpoint (mean reversion)
            if curr_close < curr_donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above Donchian midpoint (mean reversion)
            if curr_close > curr_donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals