#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 12-hour EMA trend filter and volume confirmation
# Long when price breaks above Donchian(20) upper band, 12h EMA rising, volume spike
# Short when price breaks below Donchian(20) lower band, 12h EMA falling, volume spike
# Donchian captures breakouts; 12h EMA filters for trend direction; volume confirms institutional interest
# Targets 20-50 trades/year to minimize fee drag while capturing strong moves

name = "4h_Donchian20_12hEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12-hour data once for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12-hour EMA(30) for trend filter
    close_12h = df_12h['close'].values
    ema30_12h = pd.Series(close_12h).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema30_12h)
    
    # Donchian channel (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema30_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema12h_val = ema30_12h_aligned[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian upper band, 12h EMA rising, volume spike
            if close[i] > upper_band and ema12h_val > ema30_12h_aligned[i-1] and vol_spike:
                signals[i] = 0.30
                position = 1
            # Enter short: price breaks below Donchian lower band, 12h EMA falling, volume spike
            elif close[i] < lower_band and ema12h_val < ema30_12h_aligned[i-1] and vol_spike:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian middle or 12h EMA turns down
            middle_band = (upper_band + lower_band) / 2
            if close[i] < middle_band or ema12h_val < ema30_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above Donchian middle or 12h EMA turns up
            middle_band = (upper_band + lower_band) / 2
            if close[i] > middle_band or ema12h_val > ema30_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals