#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 12h EMA50 Trend + Volume Spike
# Long when Williams %R(14) < -80 (oversold) AND price > 12h EMA50 AND volume > 2.0x 20-bar avg
# Short when Williams %R(14) > -20 (overbought) AND price < 12h EMA50 AND volume > 2.0x 20-bar avg
# Exit when Williams %R reverts to -50 (mean reversion)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 6h timeframe.
# Williams %R captures extreme reversals, 12h EMA50 filters counter-trend moves in bear markets,
# volume confirmation ensures reversal strength. Designed for both bull and bear markets via trend filter.

name = "6h_WilliamsRExtreme_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R calculation and EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h data
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R(14) on 12h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 14)  # volume MA, EMA50, and Williams %R warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50 = ema_50_12h_aligned[i]
        curr_williams_r = williams_r_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R reverts to -50 (mean reversion)
            if curr_williams_r >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R reverts to -50 (mean reversion)
            if curr_williams_r <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold) AND price > 12h EMA50 AND volume confirmation
            if curr_williams_r < -80 and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought) AND price < 12h EMA50 AND volume confirmation
            elif curr_williams_r > -20 and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals