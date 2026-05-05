#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when: Price breaks above Donchian upper (20) AND 1w EMA50 trend up AND volume > 1.5x 20-period average
# Short when: Price breaks below Donchian lower (20) AND 1w EMA50 trend down AND volume > 1.5x 20-period average
# Exit when price returns to Donchian middle (mean reversion) or opposite breakout occurs
# Donchian breakout captures volatility expansion after consolidation
# 1w EMA50 filter ensures we trade with the higher timeframe trend
# Volume confirmation reduces false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_DonchianBreakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA(50)
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian Channels (20) on 12h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_20 = (highest_20 + lowest_20) / 2.0
    
    # Calculate volume average (20) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1w EMA50 slope (up if current > previous, down if current < previous)
        if i > 100:
            ema_slope = ema_50_1w_aligned[i] - ema_50_1w_aligned[i-1]
            trend_up = ema_slope > 0
            trend_down = ema_slope < 0
        else:
            trend_up = False
            trend_down = False
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Donchian upper with uptrend and volume confirmation
            if close[i] > highest_20[i] and trend_up and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower with downtrend and volume confirmation
            elif close[i] < lowest_20[i] and trend_down and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to Donchian middle OR opposite breakout with confirmation
            if close[i] < middle_20[i] or (close[i] < lowest_20[i] and trend_down and volume_confirmed):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to Donchian middle OR opposite breakout with confirmation
            if close[i] > middle_20[i] or (close[i] > highest_20[i] and trend_up and volume_confirmed):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals