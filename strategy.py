#!/usr/bin/env python3

"""
Hypothesis: 12-hour Donchian channel breakout with daily ATR-based volatility filter and volume confirmation.
This strategy captures medium-term breakouts while filtering for low volatility environments where breakouts are more reliable.
Volume spikes confirm institutional interest. The daily ATR filter avoids choppy markets. Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate 12-hour Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12-period volume average for volume confirmation
    vol_avg_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr14_1d_aligned[i]) or np.isnan(vol_avg_12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is below its 50-period median (low volatility regime)
        if np.isnan(atr14_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Calculate rolling median of ATR for volatility regime filter
        if i >= 50:
            atr_slice = atr14_1d_aligned[max(0, i-49):i+1]
            if len(atr_slice) >= 20:
                atr_median = np.median(atr_slice[~np.isnan(atr_slice)])
                low_volatility = atr14_1d_aligned[i] < atr_median
            else:
                low_volatility = True  # Default to true if not enough data
        else:
            low_volatility = True
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume confirmation and low volatility
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * vol_avg_12[i] and
                low_volatility):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume confirmation and low volatility
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * vol_avg_12[i] and
                  low_volatility):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to the opposite Donchian level or volatility increases
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to Donchian low or volatility increases significantly
                if (close[i] < donchian_low[i] or 
                    (i >= 50 and atr14_1d_aligned[i] > 2.0 * np.median(atr14_1d_aligned[max(0, i-49):i+1][~np.isnan(atr14_1d_aligned[max(0, i-49):i+1])]))):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to Donchian high or volatility increases significantly
                if (close[i] > donchian_high[i] or 
                    (i >= 50 and atr14_1d_aligned[i] > 2.0 * np.median(atr14_1d_aligned[max(0, i-49):i+1][~np.isnan(atr14_1d_aligned[max(0, i-49):i+1])]))):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_20_1dATR_Volume_LowVol"
timeframe = "12h"
leverage = 1.0