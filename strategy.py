# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 breakouts on 12h with daily trend filter and volume confirmation
# - Camarilla R1/S1 represent key intraday support/resistance levels from previous day
# - Breakout above R1 in daily uptrend (price > EMA34) signals bullish continuation
# - Breakdown below S1 in daily downtrend (price < EMA34) signals bearish continuation
# - Volume confirmation (1.5x average) reduces false breakouts
# - Exit when price returns to pivot point or daily trend reverses
# - Uses 1d timeframe for structure and trend, 12h for execution (lower frequency = less fee drag)
# - Target: 20-40 trades/year to stay within optimal range for 12h timeframe
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Focus on BTC/ETH as primary targets to avoid overtrading pitfalls

#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Load daily data ONCE for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Camarilla pivot levels from previous day (standard formula)
    c_high = df_1d['high'].values
    c_low = df_1d['low'].values
    c_close = df_1d['close'].values
    
    pivot = (c_high + c_low + c_close) / 3
    range_val = c_high - c_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align pivot levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(c_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(pivot_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: break above R1 in daily uptrend with volume
            if close[i] > r1_12h[i] and close[i] > ema_34_12h[i] and vol_condition:
                signals[i] = 0.30
                position = 1
            # Short: break below S1 in daily downtrend with volume
            elif close[i] < s1_12h[i] and close[i] < ema_34_12h[i] and vol_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price returns to pivot or trend reverses
            if close[i] < pivot_12h[i] or close[i] < ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price returns to pivot or trend reverses
            if close[i] > pivot_12h[i] or close[i] > ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Load daily data ONCE for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Camarilla pivot levels from previous day (standard formula)
    c_high = df_1d['high'].values
    c_low = df_1d['low'].values
    c_close = df_1d['close'].values
    
    pivot = (c_high + c_low + c_close) / 3
    range_val = c_high - c_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align pivot levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(c_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(pivot_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: break above R1 in daily uptrend with volume
            if close[i] > r1_12h[i] and close[i] > ema_34_12h[i] and vol_condition:
                signals[i] = 0.30
                position = 1
            # Short: break below S1 in daily downtrend with volume
            elif close[i] < s1_12h[i] and close[i] < ema_34_12h[i] and vol_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price returns to pivot or trend reverses
            if close[i] < pivot_12h[i] or close[i] < ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price returns to pivot or trend reverses
            if close[i] > pivot_12h[i] or close[i] > ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals