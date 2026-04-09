#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Donchian breakout captures strong momentum moves in both bull and bear markets
# 1w EMA(50) filter ensures we only trade with the higher timeframe trend
# Volume confirmation (1.5x 20-day average) avoids false breakouts
# Discrete sizing 0.25 to minimize fee churn
# Target: 30-100 total trades over 4 years (7-25/year)

name = "1d_1w_donchian_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    close_s_1w = pd.Series(close_1w)
    ema_50_1w = close_s_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 1d timeframe (wait for 1w bar close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d average volume (20-period)
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        # Trend filter: price above/below 1w EMA(50)
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: breakout with volume confirmation and trend filter
            if uptrend and volume_confirmed:
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
            elif downtrend and volume_confirmed:
                if close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals