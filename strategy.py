#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA50 Trend Filter + Volume Spike
# Williams %R identifies overbought/oversold conditions. Extreme readings (<-90 or >-10) 
# combined with 1d EMA50 trend direction and volume confirmation provide high-probability 
# mean-reversion entries in ranging markets and filtered trend-following in strong trends.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 75-150 total trades over 4 years (19-38/year) to avoid fee drag.

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period) - measures overbought/oversold
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 14, 50, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i]) or
            np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_williams_r = williams_r[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Williams %R extreme and 1d EMA50 trend alignment
            if curr_volume_spike:
                # Bullish: Williams %R oversold (< -90) + price above 1d EMA50 (buy dip in uptrend)
                if curr_williams_r < -90 and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Williams %R overbought (> -10) + price below 1d EMA50 (sell rally in downtrend)
                elif curr_williams_r > -10 and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns to neutral (-50 to -20) OR loses 1d trend
            if curr_williams_r > -50 or curr_close < curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral (-80 to -50) OR loses 1d trend
            if curr_williams_r < -80 or curr_close > curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals