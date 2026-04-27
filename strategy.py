# 4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
# Hypothesis: Camarilla pivot breakouts with 12h EMA trend filter and volume confirmation
# Works in bull/bear by following 12h trend direction with breakout confirmation
# Target: 20-40 trades/year, <160 total over 4 years
# Edge: Camarilla levels identify institutional support/resistance; volume confirms institutional interest

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
    
    # Get 12h data for calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 12h ATR(14) for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h_arr, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr14_12h)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous 1d session
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_high = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_low = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Calculate R1 and S1 levels (most significant)
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 6
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Volatility filter: ATR below its 30-period median (avoid high volatility chop)
    atr_median = pd.Series(atr14_12h_aligned).rolling(window=30, min_periods=15).median().values
    vol_filter = atr14_12h_aligned < atr_median
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(atr14_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_median[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 12h uptrend + volume filter + low volatility
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema34_12h_aligned[i] and 
                volume_filter[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + 12h downtrend + volume filter + low volatility
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema34_12h_aligned[i] and 
                  volume_filter[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below S1 (support break) or trend reverses
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R1 (resistance break) or trend reverses
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0