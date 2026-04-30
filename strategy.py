#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme + Volume Spike + 12h EMA50 Trend Filter
# Williams %R identifies overbought/oversold conditions - extreme readings (< -90 or > -10) signal potential reversals
# Volume confirmation (>1.5x average) ensures institutional participation and reduces false signals
# 12h EMA50 provides higher timeframe trend filter to avoid counter-trend trades
# Works in bull/bear: extremes occur in all regimes, volume confirms legitimacy, trend filter reduces whipsaws
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "4h_WilliamsR_Extreme_VolumeSpike_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Extreme conditions: oversold (< -90) or overbought (> -10)
    williams_oversold = williams_r < -90
    williams_overbought = williams_r > -10
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_oversold = williams_oversold[i]
        curr_williams_overbought = williams_overbought[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on extreme Williams %R with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish reversal: Williams %R oversold + price above 12h EMA50
                if curr_williams_oversold and curr_close > curr_ema_50_12h:
                    signals[i] = 0.25
                    position = 1
                # Bearish reversal: Williams %R overbought + price below 12h EMA50
                elif curr_williams_overbought and curr_close < curr_ema_50_12h:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns to neutral territory (> -50) or opposite extreme
            if williams_r[i] > -50 or williams_r[i] < -90:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral territory (< -50) or opposite extreme
            if williams_r[i] < -50 or williams_r[i] > -10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals