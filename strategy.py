# 12h_PriceChannel_Momentum_Breakout
# Hypothesis: Price channel breakouts (Donchian) with momentum confirmation (ROC) and volume spike
# capture momentum bursts in both bull and bear markets. The 1d EMA200 provides trend bias,
# reducing false signals in chop. Volume > 2x average confirms institutional participation.
# ROC > 0 ensures breakout has momentum. Expected 20-30 trades/year per symbol.
name = "12h_PriceChannel_Momentum_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_band = high_series.rolling(window=20, min_periods=20).max().values
    lower_band = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    # Momentum: ROC(5) > 0
    close_series = pd.Series(close)
    roc = close_series.pct_change(5).values
    momentum = roc > 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(roc[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper band with volume spike, momentum, and above 1d EMA200
            if (close[i] > upper_band[i] and 
                volume_spike[i] and 
                momentum[i] and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume spike, negative momentum, and below 1d EMA200
            elif (close[i] < lower_band[i] and 
                  volume_spike[i] and 
                  roc[i] < 0 and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower band or breaks below 1d EMA200
            if (close[i] < lower_band[i]) or (close[i] < ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper band or breaks above 1d EMA200
            if (close[i] > upper_band[i]) or (close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals