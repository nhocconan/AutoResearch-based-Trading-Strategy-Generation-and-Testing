# Hypothesis: 6h timeframe with 1D Williams %R (oversold/overbought) + Volume Spike + Trend Filter (50 EMA)
# Williams %R identifies reversals in momentum; volume confirms strength; trend filter avoids counter-trend trades.
# Works in bull/bear by fading extremes only when aligned with higher timeframe trend.
# Target: 15-35 trades/year (60-140 over 4 years) to stay within fee-efficient range.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R (14-period) on daily
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_1d) / (highest_high - lowest_low)) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily average volume for spike detection
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Williams %R conditions: oversold (< -80) for long, overbought (> -20) for short
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: current volume above 20-period daily average volume
        vol_spike = volume[i] > vol_ma_20_1d_aligned[i]
        
        # Long conditions: oversold + uptrend + volume spike
        long_condition = oversold and price_above_ema and vol_spike
        
        # Short conditions: overbought + downtrend + volume spike
        short_condition = overbought and price_below_ema and vol_spike
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: Williams %R returns to neutral zone (-50) or trend reversal
        elif position == 1 and (williams_r_aligned[i] > -50 or not price_above_ema):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (williams_r_aligned[i] < -50 or not price_below_ema):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_Trend_Volume"
timeframe = "6h"
leverage = 1.0