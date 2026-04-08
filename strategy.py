#/usr/bin/env python3
# 4h_volume_breakout_12h_trend_v1
# Hypothesis: On 4h timeframe, breakouts above 20-bar high with volume > 2x 20-bar average and aligned with 12h EMA50 trend capture momentum moves.
# The 12h trend filter prevents counter-trend entries during ranging markets, while volume confirmation filters false breakouts.
# Works in both bull and bear markets by following the dominant trend on 12h.
# Entry: Long when price > 20-bar high AND volume > 2x 20-bar avg AND price > 12h EMA50
# Entry: Short when price < 20-bar low AND volume > 2x 20-bar avg AND price < 12h EMA50
# Exit: Opposite breakout or trend reversal
# Position sizing: 0.25 long, -0.25 short

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volume_breakout_12h_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 20-bar high/low for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 4h volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < 20-bar low OR price below 12h EMA50
            if (close[i] < low_20[i]) or (close[i] < ema_50_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price > 20-bar high OR price above 12h EMA50
            if (close[i] > high_20[i]) or (close[i] > ema_50_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price > 20-bar high + volume + price > 12h EMA50
            if (close[i] > high_20[i]) and volume_filter[i] and (close[i] > ema_50_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < 20-bar low + volume + price < 12h EMA50
            elif (close[i] < low_20[i]) and volume_filter[i] and (close[i] < ema_50_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals