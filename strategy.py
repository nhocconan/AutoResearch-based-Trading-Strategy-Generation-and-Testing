#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA50 Trend Filter and Volume Spike
# Long when Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 6h timeframe.
# Williams %R identifies exhaustion points; 1d EMA50 filters counter-trend moves; volume confirmation ensures strength.
# Works in bull via oversold bounces, in bear via overbought reversals. Novelty: Williams %R on 6h with 1d trend filter.

name = "6h_WilliamsRExtreme_1dEMA50_VolumeSpike_v1"
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
    
    # Get 14-period Williams %R on 6h data
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 50)  # volume MA, Williams %R, and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_williams_r = williams_r[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (exiting oversold territory)
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (exiting overbought territory)
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume confirmation
            if curr_williams_r < -80 and curr_close > curr_ema50_1d and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume confirmation
            elif curr_williams_r > -20 and curr_close < curr_ema50_1d and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals