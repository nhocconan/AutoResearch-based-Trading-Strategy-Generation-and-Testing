#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation.
# Long when price breaks above 10-period high AND 4h close > 4h EMA20 AND volume > 1.5x 20-period average.
# Short when price breaks below 10-period low AND 4h close < 4h EMA20 AND volume > 1.5x 20-period average.
# Uses 4h EMA20 as trend filter to avoid counter-trend trades.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 15-37 trades/year per symbol (~60-150 total over 4 years).
name = "1h_Donchian10_4hEMA20_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA20 calculation
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA20 on 4h close
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h EMA20 to 1h timeframe (wait for 4h bar close)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (10-period high/low)
    high_max_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_min_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need EMA and Donchian data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(high_max_10[i]) or np.isnan(low_min_10[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high_max = high_max_10[i]
        low_min = low_min_10[i]
        ema_trend = ema_20_4h_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above 10-period high AND above 4h EMA20
            if price > high_max and price > ema_trend and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 10-period low AND below 4h EMA20
            elif price < low_min and price < ema_trend and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below 10-period low or below 4h EMA20
            if price < low_min or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short when price breaks above 10-period high or above 4h EMA20
            if price > high_max or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals