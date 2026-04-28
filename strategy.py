#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Reversal with 1d EMA50 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions: Long when %R crosses above -80 from below,
# Short when %R crosses below -20 from above. Only trade in direction of 1d EMA50 trend.
# Volume spike (>2x 20-bar average) confirms momentum. Avoids whipsaw in ranging markets.
# Works in both bull and bear markets by only taking trend-aligned reversals from extremes.

name = "6h_WilliamsR_Reversal_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume_spike[i]
        ema_50 = ema_50_1d_aligned[i]
        wr = williams_r[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Williams %R crosses above -80 from below AND price > 1d EMA50 AND volume spike
            if i > 0 and williams_r[i-1] <= -80 and wr > -80 and curr_close > ema_50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R crosses below -20 from above AND price < 1d EMA50 AND volume spike
            elif i > 0 and williams_r[i-1] >= -20 and wr < -20 and curr_close < ema_50 and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R >= -20 (overbought) or volume drops
            if wr >= -20 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R <= -80 (oversold) or volume drops
            if wr <= -80 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals