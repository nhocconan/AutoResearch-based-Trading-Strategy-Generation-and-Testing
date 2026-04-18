#!/usr/bin/env python3
"""
1h_Donchian20_4hEMA100_Trend_1dVolSpike_Filter
Hypothesis: Trade Donchian(20) breakouts on 1h only when 4h EMA100 confirms trend and 1d volume spikes (vol > 1.5x 20-period avg). This captures strong momentum moves while filtering false breakouts. In bull markets, breakouts above upper band with rising EMA100 = long. In bear markets, breakdowns below lower band with falling EMA100 = short. Uses session filter (08-20 UTC) to avoid low-volume Asian session noise. Target: 15-30 trades/year via tight entry conditions (trend + volume + breakout).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA100 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # EMA100 on 4h
    ema_100_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 100:
        ema_100_4h[99] = np.mean(close_4h[:100])
        for i in range(100, len(close_4h)):
            ema_100_4h[i] = (close_4h[i] * 2/101) + (ema_100_4h[i-1] * 99/101)
    
    # Align EMA100 to 1h timeframe
    ema_100_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_100_4h)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 1d volume SMA20
    vol_sma_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        for i in range(20, len(volume_1d)):
            vol_sma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align 1d volume SMA to 1h timeframe
    vol_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d)
    
    # Donchian channels on 1h (20-period)
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 100, 20)  # Donchian(20), EMA100(4h), vol SMA20(1d)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_100_4h_aligned[i]) or np.isnan(vol_sma_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: 1d volume > 1.5x 20-day average
        vol_spike = volume_1d[-1] > 1.5 * vol_sma_1d_aligned[i] if len(volume_1d) > 0 else False
        
        # Trend conditions
        uptrend = close[i] > ema_100_4h_aligned[i]
        downtrend = close[i] < ema_100_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + uptrend + volume spike
            if close[i] > upper[i] and uptrend and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below lower Donchian + downtrend + volume spike
            elif close[i] < lower[i] and downtrend and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian or trend changes to down
            if close[i] < lower[i] or not uptrend:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian or trend changes to up
            if close[i] > upper[i] or not downtrend:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hEMA100_Trend_1dVolSpike_Filter"
timeframe = "1h"
leverage = 1.0