#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA50 Trend Filter + Volume Spike >2.0x
# Williams %R identifies overbought/oversold conditions (<-80 for long, >-20 for short)
# 1d EMA50 ensures alignment with primary trend; volume spike confirms institutional participation
# Discrete sizing (0.25) minimizes fee churn; target 50-150 total trades over 4 years
# Works in bull/bear: mean reversion in ranging markets, trend filter avoids counter-trend trades

name = "6h_WilliamsR_Extreme_VolumeSpike_1dEMA50_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 30, 50)  # warmup: need 14 for Williams %R, 30 for volume MA, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: Williams %R < -80 (oversold) + price above 1d EMA50
                if curr_williams_r < -80 and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R > -20 (overbought) + price below 1d EMA50
                elif curr_williams_r > -20 and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) or price below 1d EMA50
            if curr_williams_r > -20 or curr_close < curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) or price above 1d EMA50
            if curr_williams_r < -80 or curr_close > curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals