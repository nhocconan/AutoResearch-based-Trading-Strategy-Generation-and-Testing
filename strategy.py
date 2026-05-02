#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Donchian breakout captures momentum in both bull and bear markets
# 1w EMA50 ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Discrete position sizing 0.25 minimizes fee churn while maintaining exposure
# Targets 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered by 1w EMA50)

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1d volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, EMA and volume MA)
    start_idx = 60  # max(20 for Donchian, 50 for EMA, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1w EMA50
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian high AND in uptrend OR strong volume in downtrend
            if (close[i] > donchian_high[i] and 
                ((uptrend and volume_confirm[i]) or 
                 (downtrend and volume_confirm[i] and volume[i] > np.mean(volume[max(0,i-5):i]) * 2.0))):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND in downtrend OR strong volume in uptrend
            elif (close[i] < donchian_low[i] and 
                  ((downtrend and volume_confirm[i]) or 
                   (uptrend and volume_confirm[i] and volume[i] > np.mean(volume[max(0,i-5):i]) * 2.0))):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian low OR loses trend with weak volume
            if (close[i] < donchian_low[i] or 
                (not uptrend and volume[i] < np.mean(volume[max(0,i-5):i]) * 0.5)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR loses trend with weak volume
            if (close[i] > donchian_high[i] or 
                (not downtrend and volume[i] < np.mean(volume[max(0,i-5):i]) * 0.5)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals