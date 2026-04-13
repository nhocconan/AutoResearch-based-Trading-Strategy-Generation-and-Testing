#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 20-period Donchian breakout with weekly EMA200 trend filter and volume confirmation.
# Long: price breaks above Donchian upper + weekly EMA200 up + volume > 1.5x avg.
# Short: price breaks below Donchian lower + weekly EMA200 down + volume > 1.5x avg.
# Exit: opposite Donchian break or EMA200 flip.
# Designed for 1d timeframe to capture major trends with low trade frequency.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Average volume (20-period = ~1 month) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align weekly indicators to daily timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = donch_high[i]
        lower = donch_low[i]
        ema_trend = ema_200_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Donchian breakout signals
        breakout_up = price > upper
        breakout_down = price < lower
        
        # Weekly EMA200 trend direction
        ema_up = ema_200_1w_aligned[i] > ema_200_1w_aligned[i-1] if i > 0 else False
        ema_down = ema_200_1w_aligned[i] < ema_200_1w_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Long: bullish breakout + weekly EMA200 up + volume confirmation
            if (breakout_up and 
                ema_up and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: bearish breakout + weekly EMA200 down + volume confirmation
            elif (breakout_down and 
                  ema_down and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish breakout or weekly EMA200 down
            if (breakout_down or
                ema_down):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bullish breakout or weekly EMA200 up
            if (breakout_up or
                ema_up):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_20w_Donchian_EMA200_Volume"
timeframe = "1d"
leverage = 1.0