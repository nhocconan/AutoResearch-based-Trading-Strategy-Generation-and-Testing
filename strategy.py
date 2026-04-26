#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter_v1
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA20 trend filter, volume confirmation (1.8x), and UTC 08-20 session filter.
Designed for 1h timeframe to avoid overtrading by using 4h trend for direction, 1h only for precise entry timing.
Targets 15-30 trades/year/symbol by tightening volume and session filters. Works in bull/bear via 4h trend alignment.
"""

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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    # Trend: 1 = uptrend (close > EMA20), -1 = downtrend (close < EMA20), 0 = invalid
    trend_4h = np.where(ema_20_4h_aligned > 0, 
                        np.where(close > ema_20_4h_aligned, 1, -1), 
                        0)
    
    # Calculate Camarilla pivot levels from 4h OHLC (using previous 4h bar)
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 6)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 6)
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume filter: volume > 1.8 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for EMA, 20 for volume MA)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(trend_4h[i]) or not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Camarilla R1/S1 breakout conditions
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND 4h uptrend AND volume spike (1.8x)
            if close[i] > camarilla_r1_aligned[i] and trend_4h[i] == 1 and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S1 AND 4h downtrend AND volume spike (1.8x)
            elif close[i] < camarilla_s1_aligned[i] and trend_4h[i] == -1 and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: Price falls below Camarilla S1 OR 4h trend turns down
            if close[i] < camarilla_s1_aligned[i] or trend_4h[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: Price rises above Camarilla R1 OR 4h trend turns up
            if close[i] > camarilla_r1_aligned[i] or trend_4h[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0