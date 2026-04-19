#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day ATR filter and volume confirmation.
# Long when price breaks above 4h Donchian(20) high AND 1d ATR ratio > 1.5 AND volume > 1.2x 4h average volume
# Short when price breaks below 4h Donchian(20) low AND 1d ATR ratio > 1.5 AND volume > 1.2x 4h average volume
# Exit when price crosses below/above 4h Donchian(10) midpoint or when ATR ratio drops below 1.2
# ATR filter ensures trades only during high volatility regimes (breakouts, breakdowns).
# Target: 20-50 trades/year per symbol to stay within frequency limits.
name = "4h_Donchian_Breakout_ATR_Volume_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1-day ATR(14) for volatility filter
    tr1 = np.maximum(df_1d['high'], np.roll(df_1d['close'], 1)) - np.minimum(df_1d['low'], np.roll(df_1d['close'], 1))
    tr1[0] = df_1d['high'][0] - df_1d['low'][0]
    atr1 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr1_smooth = pd.Series(atr1).rolling(window=5, min_periods=1).mean().values  # smooth for stability
    atr_ratio = atr1_smooth / pd.Series(atr1_smooth).rolling(window=30, min_periods=30).mean().values
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Get 4-hour data for Donchian channels and volume average
    df_4h = get_htf_data(prices, '4h')
    
    # Donchian channels (20-period for entry, 10-period for exit)
    high_20 = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    high_10 = pd.Series(df_4h['high']).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(df_4h['low']).rolling(window=10, min_periods=10).min().values
    
    # Align Donchian channels to 4h timeframe
    donchian_high_20 = align_htf_to_ltf(prices, df_4h, high_20)
    donchian_low_20 = align_htf_to_ltf(prices, df_4h, low_20)
    donchian_high_10 = align_htf_to_ltf(prices, df_4h, high_10)
    donchian_low_10 = align_htf_to_ltf(prices, df_4h, low_10)
    
    # 4h average volume for confirmation
    vol_ma_20 = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure ATR ratio and Donchian channels are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_20 = donchian_high_20[i]
        lower_20 = donchian_low_20[i]
        mid_10 = (donchian_high_10[i] + donchian_low_10[i]) / 2
        atr_ratio_val = atr_ratio_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        vol = volume[i]
        
        # Volatility and volume confirmation
        vol_filter = atr_ratio_val > 1.5
        vol_confirmed = vol > 1.2 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above 4h Donchian(20) high AND high volatility AND volume confirmation
            if price > upper_20 and vol_filter and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 4h Donchian(20) low AND high volatility AND volume confirmation
            elif price < lower_20 and vol_filter and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 4h Donchian(10) midpoint OR volatility drops
            if price < mid_10 or atr_ratio_val < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 4h Donchian(10) midpoint OR volatility drops
            if price > mid_10 or atr_ratio_val < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals