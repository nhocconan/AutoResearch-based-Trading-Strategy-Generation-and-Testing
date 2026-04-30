#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation
# Donchian breakout captures momentum, HMA filters trend direction,
# volume spike (1.5x 20-period average) confirms strength.
# Discrete sizing 0.25 to minimize fee churn. Target: 75-200 total trades over 4 years (19-50/year).
# Works in bull via breakouts, in bear via short breakdowns + volume confirmation.

name = "4h_Donchian20_12hHMA21_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 12h HMA(21) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    hlaf_12h = pd.Series(df_12h['close'].values).rolling(window=21, min_periods=21).mean().values
    hma_12h = 2 * hlaf_12h - pd.Series(df_12h['close'].values).rolling(window=21, min_periods=21).mean().values
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_hma_12h = hma_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above Donchian high AND price > 12h HMA (uptrend)
                if curr_close > curr_donchian_high and curr_close > curr_hma_12h:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Donchian low AND price < 12h HMA (downtrend)
                elif curr_close < curr_donchian_low and curr_close < curr_hma_12h:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below Donchian low OR price drops below 12h HMA
            if curr_close < curr_donchian_low or curr_close < curr_hma_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high OR price rises above 12h HMA
            if curr_close > curr_donchian_high or curr_close > curr_hma_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals