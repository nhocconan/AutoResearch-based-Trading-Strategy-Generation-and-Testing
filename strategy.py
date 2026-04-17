#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + 1d EMA50 trend filter + volume spike confirmation
- Donchian breakouts capture momentum moves with proven edge in crypto
- 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
- Volume confirmation filters false breakouts
- Discrete position sizing (0.25) minimizes fee churn
- Target: 12-37 trades/year per symbol (~50-150 total over 4 years)
- Works in bull markets (buying upper band breakouts in uptrend) and bear markets (selling lower band breakouts in downtrend)
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
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period) on 6h primary timeframe
    # We need 20 periods of high/low for the channel
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        ema_trend = ema50_1d_aligned[i]
        vol_ma = volume_ma[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above upper Donchian band + volume spike + price > 1d EMA50 (uptrend)
            if price > upper_band and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band + volume spike + price < 1d EMA50 (downtrend)
            elif price < lower_band and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retracement to midpoint of Donchian channel
            mid_point = (upper_band + lower_band) / 2.0
            if price < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retracement to midpoint of Donchian channel
            mid_point = (upper_band + lower_band) / 2.0
            if price > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0