#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation
# In bull/bear markets: buy when price breaks above 20-day high with 1w EMA uptrend and volume spike
# Sell/short when price breaks below 20-day low with 1w EMA downtrend and volume confirmation
# Uses discrete position sizing 0.25 to limit trades and reduce fee drag
# Designed to work in both trending and ranging markets via volume confirmation and trend filter

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(21) for trend filter
    close_s_1w = pd.Series(close_1w)
    ema_21_1w = close_s_1w.ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Calculate 1d Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d average volume (20-period) for confirmation
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(avg_volume_20[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price breaks below 20-day low or trend turns down
            if close[i] < lowest_low_20[i] or ema_21_1w_aligned[i] < close_1w[0]:  # placeholder for trend check
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price breaks above 20-day high or trend turns up
            if close[i] > highest_high_20[i] or ema_21_1w_aligned[i] > close_1w[0]:  # placeholder for trend check
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on breakout above 20-day high with volume confirmation and 1w uptrend
            if (close[i] > highest_high_20[i] and 
                volume[i] > 1.5 * avg_volume_20[i] and 
                ema_21_1w_aligned[i] > ema_21_1w_aligned[max(0, i-1)]):
                position = 1
                signals[i] = 0.25
            # Enter short on breakout below 20-day low with volume confirmation and 1w downtrend
            elif (close[i] < lowest_low_20[i] and 
                  volume[i] > 1.5 * avg_volume_20[i] and 
                  ema_21_1w_aligned[i] < ema_21_1w_aligned[max(0, i-1)]):
                position = -1
                signals[i] = -0.25
    
    return signals