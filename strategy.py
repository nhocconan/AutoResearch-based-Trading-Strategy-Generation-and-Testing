#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v2
Hypothesis: On 1d timeframe, buy when price breaks above Camarilla R1 level with 1w uptrend (close > EMA34) and volume > 1.5x 20-period average; sell when price breaks below S1 level with 1w downtrend (close < EMA34) and volume > 1.5x average. Uses discrete sizing (0.0, ±0.30) to balance profit and fee drag. Targets 30-100 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # need enough for EMA34
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_R1_1d = typical_price_1d + (high_1d - low_1d) * 1.1 / 12.0
    camarilla_S1_1d = typical_price_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 1d timeframe (use previous day's levels)
    camarilla_R1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1_1d)
    camarilla_S1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1_1d)
    
    # Calculate volume average (20-period) for confirmation
    volume_s = pd.Series(volume)
    volume_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup + volume MA warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_R1_1d_aligned[i]) or 
            np.isnan(camarilla_S1_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        # 1w trend filter
        trend_uptrend = close[i] > ema_34_1w_aligned[i]
        trend_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirm + 1w uptrend
            long_signal = (close[i] > camarilla_R1_1d_aligned[i] and 
                          volume_confirm and 
                          trend_uptrend)
            
            # Short: price breaks below S1 + volume confirm + 1w downtrend
            short_signal = (close[i] < camarilla_S1_1d_aligned[i] and 
                           volume_confirm and 
                           trend_downtrend)
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: price breaks below S1 (mean reversion) OR trend change to downtrend
            if close[i] < camarilla_S1_1d_aligned[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price breaks above R1 (mean reversion) OR trend change to uptrend
            if close[i] > camarilla_R1_1d_aligned[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v2"
timeframe = "1d"
leverage = 1.0