#!/usr/bin/env python3
# 12h_donchian_breakout_daily_trend_volume_v3
# Hypothesis: Uses 12h Donchian channel (20) breakout with 1d EMA50 trend filter and volume confirmation.
# Enters long when price breaks above Donchian upper band, EMA50 rising, and volume > 1.5x average.
# Enters short when price breaks below Donchian lower band, EMA50 falling, and volume > 1.5x average.
# Exits when price re-enters the Donchian channel.
# Designed for ~20-30 trades/year on 12h to avoid fee drag. Works in bull/bear via trend-following with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_daily_trend_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 12-period Donchian channel on 12h data
    period = 20
    upperband = np.full(n, np.nan)
    lowerband = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upperband[i] = np.max(high[i - period + 1:i + 1])
        lowerband[i] = np.min(low[i - period + 1:i + 1])
    
    # 50-period EMA on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on 12h data
    vol_ma = np.full(n, np.nan)
    for i in range(20 - 1, n):
        vol_ma[i] = np.mean(volume[i - 20 + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20)  # Ensure EMA50 and Donchian are ready
    
    for i in range(start_idx, n):
        if np.isnan(upperband[i]) or np.isnan(lowerband[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x average
        vol_condition = volume[i] > 1.5 * vol_ma[i]
        
        # EMA trend: rising if current > previous, falling if current < previous
        ema_rising = ema50_1d_aligned[i] > ema50_1d_aligned[i-1]
        ema_falling = ema50_1d_aligned[i] < ema50_1d_aligned[i-1]
        
        if position == 1:  # Long position
            # Exit: price re-enters Donchian channel (closes below upper band)
            if close[i] < upperband[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters Donchian channel (closes above lower band)
            if close[i] > lowerband[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band, EMA rising, volume confirmation
            if close[i] > upperband[i] and ema_rising and vol_condition:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band, EMA falling, volume confirmation
            elif close[i] < lowerband[i] and ema_falling and vol_condition:
                position = -1
                signals[i] = -0.25
    
    return signals