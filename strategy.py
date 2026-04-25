#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolSpike
Hypothesis: Use 1h Camarilla R1/S1 breakouts with 4h EMA50 trend filter and 1d volume spike (>2.0x 20-bar MA) for entry timing. 
4h trend provides intermediate direction, 1d volume confirms institutional participation, 1h Camarilla gives precise entry/exit. 
Discrete sizing 0.20 minimizes fee drag. Target: 15-37 trades/year (~60-150 over 4 years) to stay within fee drag limits for 1h timeframe.
Works in bull/bear: 4h EMA50 adapts to trend, volume spike filters breakouts, Camarilla provides mean-reversion exit in ranging markets.
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
    open_time = prices['open_time']
    
    # Session filter: 08:00-20:00 UTC (precomputed)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume MA20 for spike detection
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate 1h Camarilla levels from previous 1h bar's OHLC
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    prev_close = prices['close'].shift(1).values
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 12   # R1 level
    s1 = prev_close - 1.1 * camarilla_range / 12   # S1 level
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 4h EMA50 (50*4=200 1h bars) and 1d volume (20 days)
    start_idx = max(200, 20*24)  # ~480 bars to be safe
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session.iloc[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND 4h trend bullish AND 1d volume spike
            long_setup = (close[i] > r1[i]) and \
                         (close[i] > ema_50_4h_aligned[i]) and \
                         (vol_spike_1d_aligned[i] > 0.5)
            # Short: price breaks below S1 AND 4h trend bearish AND 1d volume spike
            short_setup = (close[i] < s1[i]) and \
                          (close[i] < ema_50_4h_aligned[i]) and \
                          (vol_spike_1d_aligned[i] > 0.5)
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price re-enters Camarilla R1/S1 range OR 4h trend turns bearish
            if (close[i] < r1[i] and close[i] > s1[i]) or \
               (close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price re-enters Camarilla R1/S1 range OR 4h trend turns bullish
            if (close[i] < r1[i] and close[i] > s1[i]) or \
               (close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolSpike"
timeframe = "1h"
leverage = 1.0