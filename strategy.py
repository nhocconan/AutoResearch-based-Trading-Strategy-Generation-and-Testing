#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with 1-week ATR filter and volume confirmation.
Long when price breaks above 20-period Donchian high, 1-week ATR > 10-period average ATR, and volume > 20-period average volume.
Short when price breaks below 20-period Donchian low, 1-week ATR > 10-period average ATR, and volume > 20-period average volume.
Exit when price crosses Donchian median line.
Donchian breakouts capture trends; ATR filter avoids low-volatility false breakouts; volume ensures institutional participation.
Works in bull markets by catching breakouts and in bear markets by capturing breakdowns with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # 1-week ATR filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True range calculation
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1w = pd.Series(atr_1w).rolling(window=10, min_periods=10).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_1w)
    
    # Volume filter
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr_1w_aligned[i]) or np.isnan(atr_ma_1w_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, ATR above average, volume above average
            if (close[i] > donch_high[i] and 
                atr_1w_aligned[i] > atr_ma_1w_aligned[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, ATR above average, volume above average
            elif (close[i] < donch_low[i] and 
                  atr_1w_aligned[i] > atr_ma_1w_aligned[i] and 
                  volume[i] > avg_volume[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below Donchian median
                if close[i] < donch_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above Donchian median
                if close[i] > donch_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_1wATR_Volume_Filter"
timeframe = "12h"
leverage = 1.0