#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter (EMA) and volume confirmation
# Long when price breaks above 4h Donchian(20) high AND 12h EMA(20) rising AND volume > 1.5x avg
# Short when price breaks below 4h Donchian(20) low AND 12h EMA(20) falling AND volume > 1.5x avg
# Exit when price crosses 4h Donchian(20) midline OR volume drops
# Uses Donchian channel for trend-following with strict volume/EMA filters to limit trades
# Target: 75-200 total trades over 4 years, works in bull/bear via trend filter

name = "4h_donchian20_12h_ema_vol_v1"
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
    
    # 4h Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_prev = np.roll(ema_12h, 1)
    ema_12h_prev[0] = ema_12h[0]
    ema_12h_rising = ema_12h > ema_12h_prev
    ema_12h_falling = ema_12h < ema_12h_prev
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    ema_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_rising)
    ema_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_falling)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            # Long: break above Donchian high + rising 12h EMA + volume
            if (close[i] > highest_high[i] and 
                ema_12h_rising_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + falling 12h EMA + volume
            elif (close[i] < lowest_low[i] and 
                  ema_12h_falling_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals