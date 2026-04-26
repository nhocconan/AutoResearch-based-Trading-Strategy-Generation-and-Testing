#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolumeSpike
Hypothesis: On 1h timeframe, enter long when price breaks above Camarilla R1 level AND 4h trend is up (close > EMA50) AND 1d volume > 2.0x 20-period average. Enter short when price breaks below S1 level AND 4h trend is down (close < EMA50) AND 1d volume spike. Uses 4h for signal direction and 1d for volume confirmation to reduce noise. Session filter (08-20 UTC) avoids low-liquidity hours. Target: 15-35 trades/year per symbol with discrete sizing (0.20) to control fees and drawdown.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid low-liquidity periods
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla levels and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Daily Camarilla Pivot Levels (R1, S1) from previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = close + ((high-low)*1.1/12), S1 = close - ((high-low)*1.1/12)
    camarilla_r1 = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1 = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe (use previous 1d bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d Volume confirmation: volume > 2.0x 20-period average
    volume_1d = df_1d['volume'].values
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d / np.maximum(volume_ma_1d, 1e-10) > 2.0
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup (50), volume MA warmup (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Breakout conditions relative to Camarilla levels
        breakout_above_r1 = close[i] > camarilla_r1_aligned[i]
        breakout_below_s1 = close[i] < camarilla_s1_aligned[i]
        
        # 4h trend filter
        trend_uptrend = close[i] > ema_50_4h_aligned[i]
        trend_downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: price above R1 + 4h uptrend + 1d volume spike
            long_signal = breakout_above_r1 and trend_uptrend and volume_spike_1d_aligned[i]
            
            # Short: price below S1 + 4h downtrend + 1d volume spike
            short_signal = breakout_below_s1 and trend_downtrend and volume_spike_1d_aligned[i]
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price breaks below S1 OR trend change to downtrend
            if close[i] < camarilla_s1_aligned[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price breaks above R1 OR trend change to uptrend
            if close[i] > camarilla_r1_aligned[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0