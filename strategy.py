#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v2
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R1 level AND 1d trend is up (close > EMA34) AND volume > 1.5x 20-period average volume; enter short when price breaks below Camarilla S1 level AND 1d trend is down (close < EMA34) AND volume > 1.5x 20-period average volume. Uses Camarilla pivot levels from daily timeframe for structure, 1d EMA34 for trend filter, and volume confirmation to avoid false breakouts. Targets 20-50 trades per year over 4 years with discrete sizing (0.0, ±0.30) to minimize fee churn. Works in bull via trend continuation breakouts and in bear via mean reversion at extreme pivot levels.
"""

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
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need enough for EMA34
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R1, S1) from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume on 4h for volume confirmation
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (1d), EMA34 (1d), and volume MA (20) warmup
    start_idx = max(34, 20)  # EMA34 needs 34 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend + volume confirmed
            long_signal = (close[i] > camarilla_r1_aligned[i]) and trend_uptrend and volume_confirmed
            
            # Short: price breaks below S1 + 1d downtrend + volume confirmed
            short_signal = (close[i] < camarilla_s1_aligned[i]) and trend_downtrend and volume_confirmed
            
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
            # Exit: price falls below S1 OR trend change to downtrend
            if (close[i] < camarilla_s1_aligned[i]) or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price rises above R1 OR trend change to uptrend
            if (close[i] > camarilla_r1_aligned[i]) or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v2"
timeframe = "4h"
leverage = 1.0