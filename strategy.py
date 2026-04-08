#!/usr/bin/env python3
# 4h_donchian_breakout_volume_v3
# Hypothesis: Donchian breakout with volume confirmation and volatility filter. 
# Long when price breaks above 20-period Donchian high with volume > 1.5x average and ATR(14) > SMA(ATR,50).
# Short when price breaks below 20-period Donchian low with volume > 1.5x average and ATR(14) > SMA(ATR,50).
# Exit when price returns to 20-period Donchian middle or volume drops below average.
# Uses Donchian channels from 4h timeframe, volume and volatility for confirmation.
# Target: 25-50 trades/year with strict entry conditions to avoid overtrading.

import numpy as np
import pandas as pd

name = "4h_donchian_breakout_volume_v3"
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
    
    # Donchian Channel (20)
    dc_period = 20
    dc_high = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    dc_low = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    dc_middle = (dc_high + dc_low) / 2
    
    # ATR (14) for volatility filter
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # ATR SMA (50) for volatility regime filter
    atr_sma_period = 50
    atr_sma = pd.Series(atr).rolling(window=atr_sma_period, min_periods=atr_sma_period).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Volatility filter: current ATR > ATR SMA
    vol_filter = np.full(n, False)
    for i in range(n):
        if not np.isnan(atr[i]) and not np.isnan(atr_sma[i]):
            vol_filter[i] = atr[i] > atr_sma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(dc_period, vol_ma_period, atr_sma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(dc_middle[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_sma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below middle Donchian or volume drops below average
            if close[i] < dc_middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above middle Donchian or volume drops below average
            if close[i] > dc_middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above upper Donchian with volume surge and volatility filter
            if (close[i] > dc_high[i] and vol_surge[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below lower Donchian with volume surge and volatility filter
            elif (close[i] < dc_low[i] and vol_surge[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals