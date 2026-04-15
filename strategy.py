#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h volume confirmation + 12h EMA trend filter
# Uses Donchian(20) channel breakout for directional signals in both bull and bear markets.
# Long when price breaks above upper band and 12h EMA50 > 12h EMA200 (bullish trend).
# Short when price breaks below lower band and 12h EMA50 < 12h EMA200 (bearish trend).
# Volume confirmation requires current volume > 1.5x 20-bar median volume on 12h timeframe.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.
# Designed to work in trending markets (trend following) and avoid false breakouts in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # 12h volume confirmation
    vol_12h = df_12h['volume'].values
    vol_median_12h = pd.Series(vol_12h).rolling(window=20, min_periods=1).median()
    vol_threshold_12h = 1.5 * vol_median_12h
    vol_threshold_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_threshold_12h.values)
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=1).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=1).min()
    upper_band = high_20.values
    lower_band = low_20.values
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup for EMA200
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(ema200_12h_aligned[i]) or 
            np.isnan(vol_threshold_12h_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i])):
            continue
        
        # Long: price breaks above upper band, bullish 12h trend (EMA50 > EMA200), volume spike
        if (close[i] > upper_band[i] and 
            ema50_12h_aligned[i] > ema200_12h_aligned[i] and 
            volume[i] > vol_threshold_12h_aligned[i]):
            signals[i] = 0.25
        
        # Short: price breaks below lower band, bearish 12h trend (EMA50 < EMA200), volume spike
        elif (close[i] < lower_band[i] and 
              ema50_12h_aligned[i] < ema200_12h_aligned[i] and 
              volume[i] > vol_threshold_12h_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of channel or trend changes
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < (upper_band[i] + lower_band[i]) / 2 or 
                                          ema50_12h_aligned[i] < ema200_12h_aligned[i])) or
               (signals[i-1] == -0.25 and (close[i] > (upper_band[i] + lower_band[i]) / 2 or 
                                           ema50_12h_aligned[i] > ema200_12h_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0