#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_Filter_v3
Hypothesis: 4-hour Donchian channel breakouts with volume confirmation and trend filter
capture strong directional moves while avoiding chop. Uses 20-period channel for breakouts,
volume > 1.5x 20-period average for confirmation, and EMA(50) trend filter.
Designed for 15-25 trades/year to minimize fee drag. Works in bull via breakouts,
in bear via short breakdowns, and avoids whipsaws in ranging markets.
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
    
    # Donchian channel (20-period)
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    period = 20
    if len(high) >= period:
        for i in range(period, len(high)):
            donchian_high[i] = np.max(high[i-period+1:i+1])
            donchian_low[i] = np.min(low[i-period+1:i+1])
    
    # EMA(50) for trend filter
    if len(close) >= 50:
        ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50 = np.full_like(close, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 50, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and above EMA50
            if close[i] > donchian_high[i] and vol_confirm and close[i] > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and below EMA50
            elif close[i] < donchian_low[i] and vol_confirm and close[i] < ema_50[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR drops below EMA50
            if close[i] < donchian_low[i] or close[i] < ema_50[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR rises above EMA50
            if close[i] > donchian_high[i] or close[i] > ema_50[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_Filter_v3"
timeframe = "4h"
leverage = 1.0