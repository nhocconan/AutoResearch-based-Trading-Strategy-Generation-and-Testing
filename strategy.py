#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 12-hour EMA trend filter and volume confirmation
# Long when price breaks above Donchian(20) upper band with 12h EMA(50) uptrend and volume spike
# Short when price breaks below Donchian(20) lower band with 12h EMA(50) downtrend and volume spike
# Uses 4h timeframe to target 20-50 trades/year, avoiding excessive frequency
# Donchian breakouts capture momentum bursts, EMA filter ensures trend alignment, volume confirms conviction
# Exit when price reverses to opposite Donchian band or trend changes

name = "4h_Donchian20_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data once for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian(20) channels on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_12h_val = ema50_12h_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian band + 12h uptrend + volume spike
            if (not np.isnan(upper_band) and close[i] > upper_band and 
                close[i] > ema50_12h_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian band + 12h downtrend + volume spike
            elif (not np.isnan(lower_band) and close[i] < lower_band and 
                  close[i] < ema50_12h_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian band OR 12h trend turns down
            if (not np.isnan(lower_band) and close[i] < lower_band) or close[i] < ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian band OR 12h trend turns up
            if (not np.isnan(upper_band) and close[i] > upper_band) or close[i] > ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals