#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In trending markets,
# we take pullbacks: long when %R > -80 (oversold) in uptrend, short when %R < -20 (overbought) in downtrend.
# Uses 1d EMA50 for trend filter and volume spike for confirmation to avoid whipsaws.
# Target: 20-50 trades per year (80-200 total over 4 years) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Williams %R calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period Williams %R on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    williams_r = np.full(len(close_1d), -50.0)  # Initialize with neutral value
    for i in range(13, len(close_1d)):
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high - lowest_low != 0:
            williams_r[i] = -100 * (highest_high - close_1d[i]) / (highest_high - lowest_low)
        else:
            williams_r[i] = -50.0
    
    # Calculate 1-day EMA(50) for trend filter
    ema50_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (50 + 1)
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * ema_multiplier + ema50_1d[i-1]
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate average volume (12-period = 6 hours) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(12, n):
        avg_volume[i] = np.mean(volume[i-12:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(12, n):
        # Skip if any required data is not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_1d_aligned[i]
        wr = williams_r_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Williams %R > -80 (oversold) + above daily EMA50 + volume confirmation
            if (wr > -80 and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Williams %R < -20 (overbought) + below daily EMA50 + volume confirmation
            elif (wr < -20 and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R < -20 (overbought) or trend turns down
            if (wr < -20 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R > -80 (oversold) or trend turns up
            if (wr > -80 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsR_Trend_Volume"
timeframe = "4h"
leverage = 1.0