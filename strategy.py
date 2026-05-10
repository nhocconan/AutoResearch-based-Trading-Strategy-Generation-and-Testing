#!/usr/bin/env python3
# 4H_Camarilla_R4_S4_Breakout_1dEMA34_Trend_VolumeS
# Hypothesis: Uses wider Camarilla R4/S4 levels (breakout stronger than R3/S3) combined with 1d EMA34 trend filter and volume confirmation.
# Targets fewer trades (15-30/year) by requiring stronger breakouts, reducing false signals and overtrading.
# Works in bull/bear markets by aligning with 1d trend direction. Designed for BTC/ETH as primary targets.

name = "4H_Camarilla_R4_S4_Breakout_1dEMA34_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for EMA trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior day's OHLC (R4/S4 = C ± (H-L) * 1.1/2)
    camarilla_r4 = df_1d['close'] + ((df_1d['high'] - df_1d['low']) * 1.1 / 2)
    camarilla_s4 = df_1d['close'] - ((df_1d['high'] - df_1d['low']) * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (use prior 1d bar's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)
    
    # Volume filter: volume > 1.5x 34-period average on 4h chart
    vol_ma = pd.Series(volume).rolling(window=34, min_periods=34).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 34)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R4 + above 1d EMA34 + volume spike
            if (close[i] > r4_aligned[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S4 + below 1d EMA34 + volume spike
            elif (close[i] < s4_aligned[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks back below S4 (re-enters range) or volume drops below average
            if (close[i] < s4_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks back above R4 (re-enters range) or volume drops below average
            if (close[i] > r4_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals