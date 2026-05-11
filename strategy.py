#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Uses 1d Camarilla pivot levels (R1/S1) for breakout entries, with 1d EMA34 trend filter and volume confirmation.
# Works in bull markets by buying R1 breakouts in uptrends, and in bear markets by selling S1 breakdowns in downtrends.
# Designed for low trade frequency (12-37/year) to minimize fee drag.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots, EMA trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla pivot levels (R1, S1) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    tp_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # Align to 12h timeframe (wait for 1d bar to close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # --- 1d EMA34 for trend filter ---
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge_1d = df_1d['volume'].values > vol_ma_1d
    vol_surge_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_surge_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA34 (34)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_surge_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, in uptrend (price > EMA34), with volume surge
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_surge_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, in downtrend (price < EMA34), with volume surge
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_surge_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price breaks below S1 or trend turns down
                if close[i] < s1_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R1 or trend turns up
                if close[i] > r1_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals