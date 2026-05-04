#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume spike confirmation
# Donchian breakouts capture strong momentum moves. 1d ATR ensures we only trade when volatility is sufficient (>20th percentile).
# Volume spike (>1.5x 20 EMA) confirms participation. Discrete sizing 0.25 limits risk.
# Works in bull/bear: volatility filter avoids low-momentum chop. Target: 75-200 trades over 4 years.

name = "4h_Donchian20_1dATR_VolumeSpike"
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
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    tr1 = high_1d - low_1d
    tr2 = (high_1d - close_1d.shift()).abs()
    tr3 = (low_1d - close_1d.shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ATR(14) to 4h timeframe (completed 1d bar only)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate Donchian(20) channels on 4h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 20th percentile of 1d ATR for adaptive threshold (using expanding window)
    atr_percentile_20 = pd.Series(atr14_aligned).expanding(min_periods=50).quantile(0.20).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(atr14_aligned[i]) or np.isnan(atr_percentile_20[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: current 1d ATR > 20th percentile of historical ATR
        vol_filter = atr14_aligned[i] > atr_percentile_20[i]
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + vol filter + volume spike
            if close[i] > highest_high[i] and vol_filter and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + vol filter + volume spike
            elif close[i] < lowest_low[i] and vol_filter and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR volatility drops OR volume drops
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if (close[i] < midpoint or 
                not vol_filter or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR volatility drops OR volume drops
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if (close[i] > midpoint or 
                not vol_filter or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals