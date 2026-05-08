#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_DonchianBreakout_20_1dTrend_50_55"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    # Calculate 1d EMA50 and EMA55 for trend filter (1d EMA55 > EMA50 = uptrend)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_55_1d = pd.Series(close_1d).ewm(span=55, adjust=False, min_periods=55).mean().values
    trend_up_1d = ema_55_1d > ema_50_1d
    trend_down_1d = ema_55_1d < ema_50_1d
    
    # Align 1d trend to 6h
    trend_up_6h = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_6h = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # 6h Donchian channels (20-period)
    dc_up_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian and volume
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_up_20[i]) or np.isnan(dc_low_20[i]) or 
            np.isnan(trend_up_6h[i]) or np.isnan(trend_down_6h[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian upper band with volume and 1d uptrend
            long_cond = (close[i] > dc_up_20[i] and vol_filter[i] and trend_up_6h[i])
            
            # Short entry: price breaks below Donchian lower band with volume and 1d downtrend
            short_cond = (close[i] < dc_low_20[i] and vol_filter[i] and trend_down_6h[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower band (reversal signal)
            if close[i] < dc_low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper band (reversal signal)
            if close[i] > dc_up_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Donchian(20) breakout with volume confirmation and 1d EMA50/55 trend filter.
# Enters long when price breaks above 20-period Donchian high with volume surge and 1d EMA55 > EMA50 (uptrend).
# Enters short when price breaks below 20-period Donchian low with volume surge and 1d EMA55 < EMA50 (downtrend).
# Exits on reversal break of opposite Donchian band.
# Uses discrete sizing (0.25) to minimize churn. Targets 15-35 trades/year on 6h timeframe.
# Works in bull markets (trend-following breakouts) and bear markets (reversal breakouts from overextended levels).