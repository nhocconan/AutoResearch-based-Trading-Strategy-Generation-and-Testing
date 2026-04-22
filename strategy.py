#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Daily data for 20-period ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily 20-period ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    # 6h Bollinger Bands (20, 2.0)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2.0 * std20
    lower_band = sma20 - 2.0 * std20
    
    # 6h Volume surge (2x 20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(atr_20_aligned[i]) or np.isnan(sma20[i]) or np.isnan(std20[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches lower Bollinger Band + volume surge + ATR volatility filter
            if low[i] <= lower_band[i] and vol_surge[i] and atr_20_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: Price touches upper Bollinger Band + volume surge + ATR volatility filter
            elif high[i] >= upper_band[i] and vol_surge[i] and atr_20_aligned[i] > 0:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle Bollinger Band (SMA20)
            if position == 1:
                if close[i] >= sma20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] <= sma20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Bollinger_Band_Touch_VolumeSurge_ATRFilter_v1"
timeframe = "6h"
leverage = 1.0