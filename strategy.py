#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with volume confirmation and 1d EMA200 trend filter.
# Long when price breaks above 20-period Donchian high with volume > 1.5x 20-period average and price > 1d EMA200.
# Short when price breaks below 20-period Donchian low with volume > 1.5x 20-period average and price < 1d EMA200.
# Exit on opposite Donchian breakout or when price crosses 1d EMA200 in opposite direction.
# Uses 12h timeframe to reduce trade frequency (target: 15-30 trades/year) and minimize fee drag.
# Should work in both bull and bear markets by following higher timeframe trend and avoiding counter-trend trades.

name = "12h_Donchian20_1dEMA200_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for EMA200 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d EMA200 for trend direction ===
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # === 12h Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_200_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(upper) or np.isnan(lower) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high with volume confirmation and above 1d EMA200
            if close_val > upper and vol_ratio_val > 1.5 and close_val > ema_val:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with volume confirmation and below 1d EMA200
            elif close_val < lower and vol_ratio_val > 1.5 and close_val < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or crosses below 1d EMA200
            if close_val < lower or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or crosses above 1d EMA200
            if close_val > upper or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals