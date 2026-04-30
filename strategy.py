#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; extremes below -90 or above -10 signal high-probability reversals
# 1d EMA34 ensures alignment with medium-term daily trend to avoid counter-trend trades
# Volume confirmation (1.5x 50-period average) filters weak breakouts
# Discrete sizing 0.25 balances profit potential with fee minimization. Target: 80-180 total trades over 4 years (20-45/year).
# Works in bull markets via buying oversold dips in uptrends and bear markets via selling overbought rallies in downtrends.

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Williams %R (%R) with period 14
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (1.5 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # warmup for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume confirmation
            if curr_volume_confirm:
                # Bullish entry: oversold (%R < -90) and price above daily EMA (uptrend)
                if curr_williams_r < -90 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: overbought (%R > -10) and price below daily EMA (downtrend)
                elif curr_williams_r > -10 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when overbought (%R > -20) or trend changes (price below daily EMA)
            if curr_williams_r > -20 or curr_close < curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when oversold (%R < -80) or trend changes (price above daily EMA)
            if curr_williams_r < -80 or curr_close > curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals