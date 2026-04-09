#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA200 trend + volume confirmation (2x 20-period avg)
# Donchian breakouts capture strong momentum moves; EMA200 on 1d ensures we trade with higher timeframe trend
# Volume confirmation filters weak breakouts. Works in bull/bear: EMA200 trend filter avoids counter-trend whipsaws
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25-0.30

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
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR price < 1d EMA200 (trend change)
            if close[i] < donchian_low[i] or close[i] < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR price > 1d EMA200 (trend change)
            if close[i] > donchian_high[i] or close[i] > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + EMA200 trend filter
            if volume_confirmed:
                # Long entry: price > Donchian high AND price > 1d EMA200 (bullish breakout + uptrend)
                if close[i] > donchian_high[i] and close[i] > ema_200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low AND price < 1d EMA200 (bearish breakout + downtrend)
                elif close[i] < donchian_low[i] and close[i] < ema_200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals