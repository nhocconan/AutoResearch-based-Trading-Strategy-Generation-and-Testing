#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions (below -80 = oversold, above -20 = overbought).
# Combined with 12h trend (EMA50) and volume spikes, it captures mean reversion in trending markets.
# Works in both bull and bear markets by taking long signals in uptrend when oversold and short in downtrend when overbought.
# Target: 12-37 trades per year (50-150 total over 4 years) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = np.zeros(len(close_12h))
    ema_multiplier50 = 2 / (50 + 1)
    ema50_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        ema50_12h[i] = (close_12h[i] - ema50_12h[i-1]) * ema_multiplier50 + ema50_12h[i-1]
    
    # Align 12h EMA50 to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Williams %R (14-period) on 6h data
    williams_r = np.full(n, np.nan)
    for i in range(13, n):
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Calculate average volume (24-period = 4 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_12h_aligned[i]
        wr = williams_r[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + above 12h EMA50 + volume confirmation
            if (wr < -80 and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought (> -20) + below 12h EMA50 + volume confirmation
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

name = "6h_12h_WilliamsR_Trend_Volume"
timeframe = "6h"
leverage = 1.0