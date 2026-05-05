#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation
# Long when price breaks above 4h Donchian upper band AND 1d ATR(14) > 20-period average AND volume > 1.5x 20-period average
# Short when price breaks below 4h Donchian lower band AND 1d ATR(14) > 20-period average AND volume > 1.5x 20-period average
# Exit when price crosses 4h Donchian middle band (mean reversion)
# Uses 4h primary timeframe with 1d HTF for ATR volatility filter (avoid low volatility breakouts)
# Volume confirmation ensures breakouts have conviction
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Donchian20_Breakout_1dATR_Volume_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    if len(high) >= 20:
        # Upper band: highest high of last 20 periods
        upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Lower band: lowest low of last 20 periods
        lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Middle band: average of upper and lower
        middle_band = (upper_band + lower_band) / 2
    else:
        upper_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)
        middle_band = np.full(n, np.nan)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # ATR using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr[13] = np.nanmean(tr[1:15])  # First ATR value
        for i in range(14, len(tr)):
            if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            else:
                atr[i] = np.nan
    
    # Align ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # ATR filter: ATR > 20-period average (avoid low volatility breakouts)
    if len(atr_1d_aligned) >= 20:
        atr_ma_20 = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean().values
        atr_filter = atr_1d_aligned > atr_ma_20
    else:
        atr_filter = np.zeros(n, dtype=bool)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band AND ATR filter AND volume spike
            if (close[i] > upper_band[i] and 
                atr_filter[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band AND ATR filter AND volume spike
            elif (close[i] < lower_band[i] and 
                  atr_filter[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below middle band (mean reversion)
            if close[i] < middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above middle band (mean reversion)
            if close[i] > middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals