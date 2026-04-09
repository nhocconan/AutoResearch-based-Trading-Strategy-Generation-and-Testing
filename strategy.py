#!/usr/bin/env python3
# 12h_donchian_1w_volume_v1
# Hypothesis: 12h strategy using Donchian(20) breakouts with 1-week EMA trend filter and volume confirmation.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to avoid fee drag.
# Works in bull/bear by using 1-week EMA trend filter and requiring volume spike for confirmation.
# Uses discrete sizing (±0.25) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_1w_volume_v1"
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
    
    # 1w HTF data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian channels (20-period) using available data up to i
        lookback = min(20, i+1)
        highest_high = np.max(high[i-lookback+1:i+1])
        lowest_low = np.min(low[i-lookback+1:i+1])
        
        # Volume confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            volume_ma = np.mean(volume[i-20:i])
            volume_confirmed = volume[i] > 2.0 * volume_ma
        else:
            volume_confirmed = False
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band OR trend reversal
            if close[i] <= lowest_low or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band OR trend reversal
            if close[i] >= highest_high or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long conditions: price breaks above upper Donchian band + uptrend
                if close[i] >= highest_high and close[i] > ema_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short conditions: price breaks below lower Donchian band + downtrend
                elif close[i] <= lowest_low and close[i] < ema_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals