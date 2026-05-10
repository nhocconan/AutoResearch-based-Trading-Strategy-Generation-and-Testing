#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Filter_1dTrend_VolumeBreakout
# Hypothesis: Ichimoku Cloud (TK Cross + Cloud twist) on 6h with 1d EMA trend filter and volume breakout.
# In bull: price above cloud + TK cross up + 1d uptrend + volume spike = long.
# In bear: price below cloud + TK cross down + 1d downtrend + volume spike = short.
# Cloud acts as dynamic support/resistance; TK cross signals momentum shift.
# Works in both regimes by aligning with higher timeframe trend. Targets 15-30 trades/year.

name = "6h_Ichimoku_Cloud_Filter_1dTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components (9, 26, 52)
    tenkan = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
              pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
             pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (24-period MA on 6h)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(9, 26, 52, 50, 24) + 26  # Warmup for Ichimoku + daily EMA + volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ichimoku signals
        tk_cross = tenkan[i] > kijun[i]  # bullish TK cross
        price_above_cloud = close[i] > max(senkou_a[i], senkou_b[i])
        price_below_cloud = close[i] < min(senkou_a[i], senkou_b[i])
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: price above cloud + TK cross up + 1d uptrend + volume spike
            if price_above_cloud and tk_cross and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price below cloud + TK cross down + 1d downtrend + volume spike
            elif price_below_cloud and not tk_cross and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below cloud or TK cross down
            if price_below_cloud or not tk_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above cloud or TK cross up
            if price_above_cloud or tk_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals