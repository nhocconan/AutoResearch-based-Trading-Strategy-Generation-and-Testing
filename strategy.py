#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter.
# Long: Williams %R(14) < -80 (oversold) + 12h EMA20 trend up + volume > 1.2x average.
# Short: Williams %R(14) > -20 (overbought) + 12h EMA20 trend down + volume > 1.2x average.
# Uses mean reversion in ranging markets with trend alignment to avoid counter-trend trades.
# Williams %R identifies exhaustion points; 12h EMA20 filters for trend direction.
# Volume confirmation ensures participation. Designed for 60-120 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) on 6h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(13, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    
    williams_r = np.full(n, np.nan)
    for i in range(13, n):
        hh = highest_high[i]
        ll = lowest_low[i]
        if hh != ll:
            williams_r[i] = -100 * (hh - close[i]) / (hh - ll)
        else:
            williams_r[i] = -50  # avoid division by zero
    
    # 12h data for trend filter (EMA20)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = np.full(len(close_12h), np.nan)
    for i in range(19, len(close_12h)):
        if i == 19:
            ema_12h[i] = np.mean(close_12h[:20])
        else:
            ema_12h[i] = close_12h[i] * 2 / (20 + 1) + ema_12h[i-1] * (19 / (20 + 1))
    
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        ema_val = ema_12h_aligned[i]
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 1.2x average volume
        volume_confirm = vol > 1.2 * avg_vol
        
        # Trend filter: price above/below 12h EMA20
        trend_up = price > ema_val
        trend_down = price < ema_val
        
        if position == 0:
            # Long: oversold + uptrend + volume
            if (wr < -80 and trend_up and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: overbought + downtrend + volume
            elif (wr > -20 and trend_down and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (-50) or trend breaks
            if wr > -50 or not trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (-50) or trend breaks
            if wr < -50 or not trend_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_WilliamsR_MeanReversion_Trend"
timeframe = "6h"
leverage = 1.0