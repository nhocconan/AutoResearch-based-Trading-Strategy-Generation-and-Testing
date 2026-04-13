#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean-reversion with weekly trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions (-20 to -80 range).
# Weekly trend filter ensures trades align with higher timeframe direction.
# Volume confirmation avoids false signals in low liquidity.
# Target: 15-30 trades per year (60-120 total over 4 years) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Williams %R(14)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(14, n):
        highest_high[i] = np.max(high[i-14:i+1])
        lowest_low[i] = np.min(low[i-14:i+1])
    
    williams_r = np.full(n, np.nan)
    for i in range(14, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (20 + 1)
    ema_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_1w[i] = (close_1w[i] - ema_1w[i-1]) * ema_multiplier + ema_1w[i-1]
    
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        wr = williams_r[i]
        weekly_ema = ema_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.2x average volume
        volume_confirm = vol > 1.2 * avg_vol
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + above weekly EMA + volume confirmation
            if (wr < -80 and
                price > weekly_ema and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Williams %R > -20 (overbought) + below weekly EMA + volume confirmation
            elif (wr > -20 and
                  price < weekly_ema and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R > -50 (return from oversold)
            if wr > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R < -50 (return from overbought)
            if wr < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WilliamsR_MeanReversion_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0