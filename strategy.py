#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above upper Donchian channel (20) AND price > 1w EMA50 (strong uptrend) AND volume spike
# Short when price breaks below lower Donchian channel (20) AND price < 1w EMA50 (strong downtrend) AND volume spike
# Donchian channels provide clear trend-following structure with defined breakout levels
# 1w EMA50 offers a smoother, longer-term trend filter to reduce whipsaw in ranging markets
# Volume spike (2.0x 20-bar MA) confirms institutional participation in breakouts
# Works in bull markets (trend continuation) and bear markets (sharp reversals on volume)
# Timeframe: 1d (primary timeframe as required)

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to insufficient data for EMA)
        if np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels from lookback 20 periods (excluding current bar)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            # Not enough lookback data
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 2.0x 20-bar moving average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_spike = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_spike = False
        
        if position == 0:
            # Long: price breaks above upper Donchian AND strong uptrend AND volume spike
            if (close[i] > highest_high and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND strong downtrend AND volume spike
            elif (close[i] < lowest_low and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below upper Donchian OR below 1w EMA50
            if close[i] < highest_high or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above lower Donchian OR above 1w EMA50
            if close[i] > lowest_low or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals