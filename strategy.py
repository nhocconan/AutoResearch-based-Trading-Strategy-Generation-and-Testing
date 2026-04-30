#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R extreme with 1d EMA50 trend filter and volume confirmation (>1.8x average)
# Uses 12h timeframe to reduce trade frequency (target: 50-150 total trades over 4 years)
# Williams %R identifies oversold/overbought conditions: long when %R < -80, short when %R > -20
# 1d EMA50 provides trend filter to avoid counter-trend trades
# Volume confirmation >1.8x 20-period average ensures breakout legitimacy
# Discrete position sizing: 0.25 for entries to limit fee drag
# Works in all markets: mean reversion in ranging markets, trend filter prevents false signals in trends

name = "12h_WilliamsR_Extreme_1dEMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams %R (14-period) using previous bar to avoid look-ahead
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using previous bar's data for calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate rolling max/min for Williams %R
    highest_high = pd.Series(prev_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(prev_low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R formula
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - prev_close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when no range
    )
    
    # Extreme conditions: oversold (< -80) and overbought (> -20)
    oversold = williams_r < -80
    overbought = williams_r > -20
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 14, 20, 50)  # warmup for Williams %R (14), volume MA (20), EMA (50)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on Williams %R extreme with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish: oversold (%R < -80) + price above 1d EMA50
                if curr_williams_r < -80 and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish: overbought (%R > -20) + price below 1d EMA50
                elif curr_williams_r > -20 and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns to neutral range (-50) or opposite extreme
            if curr_williams_r > -50:  # exited oversold condition
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral range (-50) or opposite extreme
            if curr_williams_r < -50:  # exited overbought condition
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals