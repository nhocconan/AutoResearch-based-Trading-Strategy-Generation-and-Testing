#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation.
# Donchian(20) identifies breakouts of recent price channels.
# 1d EMA200 filters for higher-timeframe trend to avoid counter-trend trades.
# Volume > 1.5x 20-period average confirms institutional participation.
# This structure captures trends in both bull and bear markets while avoiding range-bound periods.
# Target: 20-40 trades per year to minimize fee drag.

name = "4h_Donchian20_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for EMA200 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d EMA200 for trend direction ===
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # === 4h Donchian(20) channel ===
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 4h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after Donchian warmup
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
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price above Donchian high + uptrend filter + volume
            if close_val > upper and close_val > ema_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short breakdown: price below Donchian low + downtrend filter + volume
            elif close_val < lower and close_val < ema_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: price closes below Donchian low or trend reversal
            if close_val < lower or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above Donchian high or trend reversal
            if close_val > upper or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals