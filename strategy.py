#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d volume confirmation and ATR volatility filter.
# Long when price breaks above 20-period Donchian high with above-average volume and ATR > 20-period MA.
# Short when price breaks below 20-period Donchian low with above-average volume and ATR > 20-period MA.
# Uses daily volume filter to avoid low-conviction breakouts. ATR filter ensures sufficient volatility.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range.

name = "12h_donchian20_1d_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Average True Range (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Daily volume filter: volume above 20-period average
    df_1d = get_htf_data(prices, '1d')
    daily_volume = df_1d['volume'].values
    daily_vol_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    daily_vol_above_avg = daily_volume > daily_vol_ma
    daily_vol_above_avg_aligned = align_htf_to_ltf(prices, df_1d, daily_vol_above_avg)
    
    # Daily ATR filter: ATR above 20-period MA
    df_1d_atr = get_htf_data(prices, '1d')
    high_1d = df_1d_atr['high'].values
    low_1d = df_1d_atr['low'].values
    close_1d = df_1d_atr['close'].values
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_above_ma_1d = atr_1d > atr_ma_1d
    atr_above_ma_1d_aligned = align_htf_to_ltf(prices, df_1d_atr, atr_above_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily filter data not available
        if np.isnan(daily_vol_above_avg_aligned[i]) or np.isnan(atr_above_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses opposite Donchian band
        if position == 1:  # long position
            if close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and ATR filters
            # Long: price breaks above Donchian high with volume and ATR confirmation
            if (close[i] > donch_high[i] and 
                daily_vol_above_avg_aligned[i] and 
                atr_above_ma_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and ATR confirmation
            elif (close[i] < donch_low[i] and 
                  daily_vol_above_avg_aligned[i] and 
                  atr_above_ma_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals