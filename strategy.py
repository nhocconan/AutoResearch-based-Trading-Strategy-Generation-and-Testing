#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + Volume Spike + 1d EMA50 Trend Filter
# Williams %R identifies overbought/oversold conditions: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Extreme readings: %R < -80 (oversold) for longs, %R > -20 (overbought) for shorts
# Volume confirmation (>2.0x 30-period average) ensures institutional participation
# 1d EMA50 trend filter avoids counter-trend trades during strong trends
# Works in bull/bear: volume confirms breakouts, 1d EMA50 filters whipsaws, Williams %R catches reversals
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_WilliamsR_Extreme_VolumeSpike_1dEMA50_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(30, 14, 50)  # warmup for volume MA, Williams %R, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma_30[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: Williams %R < -80 (oversold) and price above 1d EMA50
                if curr_williams_r < -80 and curr_close > curr_ema_50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Williams %R > -20 (overbought) and price below 1d EMA50
                elif curr_williams_r > -20 and curr_close < curr_ema_50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Williams %R rises above -50 (momentum fading) or price breaks below 1d EMA50
            if curr_williams_r > -50 or curr_close < curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -50 (momentum fading) or price breaks above 1d EMA50
            if curr_williams_r < -50 or curr_close > curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals