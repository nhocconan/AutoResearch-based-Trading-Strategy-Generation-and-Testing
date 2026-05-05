#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when: Price breaks above 20-period Donchian high (1d) AND price > 1w EMA50 AND 1d volume > 1.8x 20-period average
# Short when: Price breaks below 20-period Donchian low (1d) AND price < 1w EMA50 AND 1d volume > 1.8x 20-period average
# Exit when price touches opposite Donchian level (20-period low for long, high for short)
# Donchian channels provide clear trend-following structure with defined risk
# 1w EMA50 filters for primary trend alignment (avoid counter-trend trades)
# Volume spike confirms institutional participation and reduces false breakouts
# Target: 30-80 total trades over 4 years (7-20/year) with discrete sizing 0.25

name = "1d_Donchian20_1wEMA50_VolumeSpike_1.8x"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop (primary timeframe data is already aligned)
    df_1d = prices  # Since timeframe is 1d, prices is already 1d data
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    # Upper band = highest high over last 20 periods
    # Lower band = lowest low over last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume spike (current volume > 1.8x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period for 20-period indicators
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        vol_cond = bool(volume[i] > (1.8 * vol_ma_20[i])) if not np.isnan(vol_ma_20[i]) else False
        above_ema = close[i] > ema_50_1w_aligned[i] if not np.isnan(ema_50_1w_aligned[i]) else False
        below_ema = close[i] < ema_50_1w_aligned[i] if not np.isnan(ema_50_1w_aligned[i]) else False
        
        if position == 0:
            # Long: Break above Donchian high AND above 1w EMA50 AND volume spike
            if close[i] > highest_high[i] and above_ema and vol_cond:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND below 1w EMA50 AND volume spike
            elif close[i] < lowest_low[i] and below_ema and vol_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: touch Donchian low (opposite side)
            if close[i] <= lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: touch Donchian high (opposite side)
            if close[i] >= highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals