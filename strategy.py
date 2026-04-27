#!/usr/bin/env python3
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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d ATR(20) for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr20_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr20_1d)
    
    # Volume filter: volume > 1.8x 20-period average (strong volume)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    # Volatility filter: ATR below its 50-period median (low volatility regime)
    atr_median = pd.Series(atr20_1d_aligned).rolling(window=50, min_periods=20).median().values
    vol_filter = atr20_1d_aligned < atr_median
    
    # Bollinger Band width filter (range detection)
    bb_middle = pd.Series(close_1d_arr).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d_arr).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    bb_width_ma = pd.Series(bb_width_aligned).rolling(window=20, min_periods=20).mean().values
    range_filter = bb_width_aligned < bb_width_ma  # narrow bands = ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr20_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_median[i]) or np.isnan(bb_width_aligned[i]) or
            np.isnan(bb_width_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above EMA50 + strong volume + low volatility + ranging market
            if (close[i] > ema50_1d_aligned[i] and 
                volume_filter[i] and 
                vol_filter[i] and
                range_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below EMA50 + strong volume + low volatility + ranging market
            elif (close[i] < ema50_1d_aligned[i] and 
                  volume_filter[i] and 
                  vol_filter[i] and
                  range_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below EMA50 (trend change) OR volatility increases
            if close[i] < ema50_1d_aligned[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above EMA50 (trend change) OR volatility increases
            if close[i] > ema50_1d_aligned[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA50_Vol_LowVol_RangeFilter_v2"
timeframe = "4h"
leverage = 1.0