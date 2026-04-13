#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels, effective in ranging markets.
# Combined with 1d EMA trend filter and volume spikes, it avoids whipsaws and trades with momentum.
# Works in both bull and bear markets by taking long signals only in uptrend and short in downtrend.
# Target: 12-37 trades per year (50-150 total over 4 years) for 12h timeframe.

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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = np.full(len(high_1d), np.nan)
    lowest_low = np.full(len(low_1d), np.nan)
    
    for i in range(14, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-14:i+1])
        lowest_low[i] = np.min(low_1d[i-14:i+1])
    
    williams_r = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i]) * -100
    
    # Calculate 1-day EMA(50) for trend filter
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * (2 / (50 + 1)) + ema50_1d[i-1]
    
    # Align all indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate average volume (2-period = 1 day) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(2, n):
        avg_volume[i] = np.mean(volume[i-2:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(2, n):
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
            # Long: Williams %R < -80 (oversold) + above daily EMA50 + volume confirmation
            if (wr < -80 and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Williams %R > -20 (overbought) + below daily EMA50 + volume confirmation
            elif (wr > -20 and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R > -50 or trend turns down
            if (wr > -50 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R < -50 or trend turns up
            if (wr < -50 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_WilliamsR_Trend_Volume"
timeframe = "12h"
leverage = 1.0