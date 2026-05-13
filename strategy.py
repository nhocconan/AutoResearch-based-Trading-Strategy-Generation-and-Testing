#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Squeeze"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    # P = (H+L+C)/3
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 4H timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1D EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 20-period average
    volume_s = pd.Series(volume)
    vol_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    # Bollinger Bands width for squeeze detection (20, 2)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean()
    bb_std = close_s.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_ma50 = bb_width.rolling(window=50, min_periods=50).mean()
    squeeze_condition = bb_width < bb_width_ma50  # Bollinger Squeeze
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(squeeze_condition[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to EMA34
        price_above_ema = close[i] > ema34_1d_aligned[i]
        price_below_ema = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # LONG: Break above R1 with volume, uptrend, and squeeze
            if (close[i] > r1_aligned[i]) and price_above_ema and volume_ok[i] and squeeze_condition[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume, downtrend, and squeeze
            elif (close[i] < s1_aligned[i]) and price_below_ema and volume_ok[i] and squeeze_condition[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below S1 or squeeze breaks
            if (close[i] < s1_aligned[i]) or not squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above R1 or squeeze breaks
            if (close[i] > r1_aligned[i]) or not squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals