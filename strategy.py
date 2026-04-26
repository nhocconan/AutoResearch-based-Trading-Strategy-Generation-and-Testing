#!/usr/bin/env python3
"""
4h_DonchianBreakout_VolumeConfirm_12hTrend_v1
Hypothesis: On 4h timeframe, buy when price breaks above 20-period Donchian high with volume > 1.8x 20-period average and 12h uptrend (close > EMA50); sell when price breaks below 20-period Donchian low with volume > 1.8x average and 12h downtrend. Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 20-50 trades per year over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period) for confirmation
    volume_s = pd.Series(volume)
    volume_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian warmup + EMA warmup + volume MA warmup
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirm = volume[i] > 1.8 * volume_ma_20[i]
        
        # 12h trend filter
        trend_uptrend = close[i] > ema_50_12h_aligned[i]
        trend_downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume confirm + 12h uptrend
            long_signal = (close[i] > donchian_high[i] and 
                          volume_confirm and 
                          trend_uptrend)
            
            # Short: price breaks below Donchian low + volume confirm + 12h downtrend
            short_signal = (close[i] < donchian_low[i] and 
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
            # Exit: price breaks below Donchian low (mean reversion) OR trend change to downtrend
            if close[i] < donchian_low[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian high (mean reversion) OR trend change to uptrend
            if close[i] > donchian_high[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_DonchianBreakout_VolumeConfirm_12hTrend_v1"
timeframe = "4h"
leverage = 1.0