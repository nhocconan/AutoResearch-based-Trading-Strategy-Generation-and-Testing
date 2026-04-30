#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
# Long when price breaks above upper Donchian channel AND 1d ATR(14) > 20-bar median ATR AND volume > 1.5x 20-bar average volume.
# Short when price breaks below lower Donchian channel AND 1d ATR(14) > 20-bar median ATR AND volume > 1.5x 20-bar average volume.
# Exit when price crosses the middle Donchian channel (20-bar midpoint).
# Uses discrete position sizing (0.25) to reduce fee churn and manage drawdown.
# Target: 75-150 total trades over 4 years (19-38/year) for 4h timeframe.
# Works in bull/bear via volatility filter that avoids low-volume false breakouts and adapts to changing market conditions.

name = "4h_Donchian20_1dATR_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian channels from 20-period high/low
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_channel = high_ma_20
    lower_channel = low_ma_20
    middle_channel = (upper_channel + lower_channel) / 2.0
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Volatility filter: 1d ATR(14) > 20-bar median of ATR (adaptive threshold)
    atr_median_20 = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).median().values
    volatility_filter = atr_14_1d_aligned > atr_median_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_median_20[i]) or
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(middle_channel[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_volatility_filter = volatility_filter[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above upper channel, volatility filter passed, volume confirmation
            if (curr_high > upper_channel[i] and 
                curr_volatility_filter and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel, volatility filter passed, volume confirmation
            elif (curr_low < lower_channel[i] and 
                  curr_volatility_filter and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below middle channel
            if curr_close < middle_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above middle channel
            if curr_close > middle_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals