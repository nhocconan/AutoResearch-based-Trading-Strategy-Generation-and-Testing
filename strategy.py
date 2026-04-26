#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: On 4h timeframe, buy when price breaks above Camarilla R1 level with 12h uptrend (close > EMA50) and volume > 2.0x 20-period average; sell when price breaks below S1 level with 12h downtrend (close < EMA50) and volume > 2.0x average. Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 20-50 trades per year over 4 years. Focus on BTC/ETH edge via tight volume confirmation and HTF trend filter.
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
    
    # Get 12h data for HTF trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # need enough for EMA50
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla calculation (more stable than lower TF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # need at least 2 days for previous day levels
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_R1_1d = typical_price_1d + (high_1d - low_1d) * 1.1 / 12.0
    camarilla_S1_1d = typical_price_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_R1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1_1d)
    camarilla_S1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1_1d)
    
    # Calculate volume average (20-period) for confirmation
    volume_s = pd.Series(volume)
    volume_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup + volume MA warmup
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_R1_1d_aligned[i]) or 
            np.isnan(camarilla_S1_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average (tight filter)
        volume_confirm = volume[i] > 2.0 * volume_ma_20[i]
        
        # 12h trend filter
        trend_uptrend = close[i] > ema_50_12h_aligned[i]
        trend_downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirm + 12h uptrend
            long_signal = (close[i] > camarilla_R1_1d_aligned[i] and 
                          volume_confirm and 
                          trend_uptrend)
            
            # Short: price breaks below S1 + volume confirm + 12h downtrend
            short_signal = (close[i] < camarilla_S1_1d_aligned[i] and 
                           volume_confirm and 
                           trend_downtrend)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S1 (mean reversion) OR trend change to downtrend
            if close[i] < camarilla_S1_1d_aligned[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 (mean reversion) OR trend change to uptrend
            if close[i] > camarilla_R1_1d_aligned[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0