#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume spike and 12h EMA50 trend filter
# Uses proven price channel breakout structure with volume confirmation and multi-timeframe trend filter
# Works in bull/bear: volume spike confirms institutional interest, EMA50 filters counter-trend noise
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Donchian20_VolumeSpike_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # 12h EMA50 for trend filter (MTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50)  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: price breaks above upper Donchian with volume and above 12h EMA50
                if curr_high > high_20[i] and curr_close > curr_ema_50:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below lower Donchian with volume and below 12h EMA50
                elif curr_low < low_20[i] and curr_close < curr_ema_50:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below lower Donchian (reversal signal)
            if curr_low < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit when price breaks above upper Donchian (reversal signal)
            if curr_high > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals