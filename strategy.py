#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1w trend filter and volume confirmation.
# Long when price breaks above upper Donchian(20) with price above 1w EMA50 and volume spike (>1.6x average).
# Short when price breaks below lower Donchian(20) with price below 1w EMA50 and volume spike.
# Uses 1w EMA50 as trend filter to avoid counter-trend trades, reducing whipsaw in sideways markets.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years).
name = "4h_Donchian20_1wEMA50_Volume"
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
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on weekly close
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 4h timeframe (wait for weekly close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.6x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Need Donchian and EMA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_donchian = high_max_20[i]
        lower_donchian = low_min_20[i]
        ema_trend = ema_50_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.6 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above upper Donchian AND above 1w EMA50
            if price > upper_donchian and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian AND below 1w EMA50
            elif price < lower_donchian and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below lower Donchian or below 1w EMA50
            if price < lower_donchian or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above upper Donchian or above 1w EMA50
            if price > upper_donchian or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals