#!/usr/bin/env python3
# 4h_donchian20_volume_sr_pullback_v1
# Hypothesis: Buy pullbacks to Donchian support/resistance with volume confirmation in 4h timeframe.
# Long when price touches or crosses above lower Donchian band (20-period) with volume > 1.5x average.
# Short when price touches or crosses below upper Donchian band with volume > 1.5x average.
# Exit when price crosses the middle (midpoint of Donchian channel) or volume drops below average.
# Uses 12h timeframe for trend filter: only take longs when 12h trend is up (price > 12h EMA50), shorts when down.
# Target: 20-40 trades/year with strict entry conditions to avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_volume_sr_pullback_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donchian_period-1, n):
        upper[i] = np.max(high[i-donchian_period+1:i+1])
        lower[i] = np.min(low[i-donchian_period+1:i+1])
    middle = (upper + lower) / 2.0
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donchian_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below middle or volume drops below average
            if close[i] < middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above middle or volume drops below average
            if close[i] > middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price touches/crosses above lower band with volume surge and 12h uptrend
            if (close[i] >= lower[i] and close[i-1] < lower[i-1]) and vol_surge[i] and (close[i] > ema_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches/crosses below upper band with volume surge and 12h downtrend
            elif (close[i] <= upper[i] and close[i-1] > upper[i-1]) and vol_surge[i] and (close[i] < ema_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals