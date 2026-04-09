#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA200 trend filter + volume confirmation
# Donchian breakouts capture strong momentum moves; EMA200 ensures we trade with higher timeframe trend
# Volume confirmation (1.5x 20-period avg) filters weak breakouts
# Works in bull/bear: EMA200 trend filter avoids counter-trend whipsaws in ranging markets
# Discrete sizing 0.25 to control risk and minimize fee churn
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_1d_donchian_ema200_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend direction
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.max(high[i-20:i])
            lowest_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian lower band OR price < 1d EMA200 (trend change)
            if close[i] < lowest_low[i] or close[i] < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian upper band OR price > 1d EMA200 (trend change)
            if close[i] > highest_high[i] or close[i] > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + EMA200 trend filter
            if volume_confirmed:
                # Long entry: price > Donchian upper band AND price > 1d EMA200 (bullish breakout + uptrend)
                if close[i] > highest_high[i] and close[i] > ema_200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian lower band AND price < 1d EMA200 (bearish breakout + downtrend)
                elif close[i] < lowest_low[i] and close[i] < ema_200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals