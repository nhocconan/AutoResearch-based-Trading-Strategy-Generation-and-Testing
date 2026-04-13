#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d trend filter and volume confirmation.
# Williams %R measures overbought/oversold conditions; combined with 1d EMA trend and volume spikes,
# it captures mean reversion in trending markets while avoiding whipsaws.
# Works in both bull and bear markets by taking long signals only in uptrend (oversold bounces)
# and short in downtrend (overbought reversals). Target: 20-50 trades per year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.zeros(len(close_1d))
    ema_multiplier50 = 2 / (50 + 1)
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * ema_multiplier50 + ema50_1d[i-1]
    
    # Calculate 14-period Williams %R on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    williams_r = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Align indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate average volume (28-period = 7 hours) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(28, n):
        avg_volume[i] = np.mean(volume[i-28:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(28, n):
        # Skip if any required data is not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_1d_aligned[i]
        wr = williams_r_aligned[i]
        
        # Volume confirmation: current volume > 1.8x average volume
        volume_confirm = vol > 1.8 * avg_vol
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + above daily EMA50 + volume confirmation
            if (wr < -80 and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought (> -20) + below daily EMA50 + volume confirmation
            elif (wr > -20 and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R rises above -50 or trend turns down
            if (wr > -50 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R falls below -50 or trend turns up
            if (wr < -50 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsR_Trend_Volume"
timeframe = "4h"
leverage = 1.0