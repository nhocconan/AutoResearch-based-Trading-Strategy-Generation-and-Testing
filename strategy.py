#!/usr/bin/env python3
# 4h_1d_cci_trend_volume_v1
# Hypothesis: Trade CCI(20) overbought/oversold reversals on 4h timeframe with daily trend filter and volume confirmation.
# In bullish regime (price > 50-day SMA): long when CCI crosses below -100 (oversold bounce).
# In bearish regime (price < 50-day SMA): short when CCI crosses above +100 (overbought rejection).
# Uses volume > 1.3x 20-period average to confirm momentum shift.
# Designed for 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years).
# Works in both bull and bear markets by adapting to trend via daily SMA filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily 50-period SMA for trend filter
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # CCI(20) calculation on 4h data
    tp = (high + low + close) / 3.0  # Typical Price
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(np.abs(tp - ma_tp)).rolling(window=20, min_periods=20).mean().values
    # Avoid division by zero
    cci = np.where(mad != 0, (tp - ma_tp) / (0.015 * mad), 0.0)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma50_1d_aligned[i]) or np.isnan(cci[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ma_tp[i]) or np.isnan(mad[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.3 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: CCI crosses above +100 (overbought) or trend turns bearish
            if cci[i] > 100.0 or close[i] < sma50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses below -100 (oversold) or trend turns bullish
            if cci[i] < -100.0 or close[i] > sma50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: CCI crosses below -100 (oversold bounce) with volume surge and bullish trend
            if (cci[i] <= -100.0 and cci[i-1] > -100.0 and vol_surge and 
                close[i] > sma50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: CCI crosses above +100 (overbought rejection) with volume surge and bearish trend
            elif (cci[i] >= 100.0 and cci[i-1] < 100.0 and vol_surge and 
                  close[i] < sma50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals