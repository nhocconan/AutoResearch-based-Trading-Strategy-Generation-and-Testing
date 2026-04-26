#!/usr/bin/env python3
"""
1h_HighLowBreakout_VolumeSpike_4hTrend
Hypothesis: 1h breakout above recent 20-period high or below recent 20-period low with volume spike and 4h trend filter.
- Long when price breaks above highest(high,20) AND volume > 2.0 * volume_ma(20) AND 4h close > 4h EMA50
- Short when price breaks below lowest(low,20) AND volume > 2.0 * volume_ma(20) AND 4h close < 4h EMA50
- Uses discrete position sizing (0.20) to minimize fee churn
- Volume spike confirms institutional participation
- 4h EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Designed for moderate frequency (target 15-35 trades/year on 1h) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter (needs completed 4h candle)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_4h = np.where(ema_50_4h_aligned > 0, 
                        np.where(close > ema_50_4h_aligned, 1, -1), 
                        0)
    
    # Calculate highest high and lowest low for breakout levels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume filter: volume > 2.0 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 4h EMA, 20 for breakout levels and volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(trend_4h[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above 20-period high AND 4h uptrend AND volume spike
            if close[i] > highest_high[i] and trend_4h[i] == 1 and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 20-period low AND 4h downtrend AND volume spike
            elif close[i] < lowest_low[i] and trend_4h[i] == -1 and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: Price falls below 20-period low OR 4h trend turns down
            if close[i] < lowest_low[i] or trend_4h[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: Price rises above 20-period high OR 4h trend turns up
            if close[i] > highest_high[i] or trend_4h[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_HighLowBreakout_VolumeSpike_4hTrend"
timeframe = "1h"
leverage = 1.0