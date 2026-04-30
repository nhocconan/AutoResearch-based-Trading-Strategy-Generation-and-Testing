#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 12h EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions (-20 to -80 range)
# Extreme readings below -80 (oversold) or above -20 (overbought) signal potential reversals
# Entry on reversal from extreme with volume confirmation and 12h EMA50 trend alignment
# Works in bull markets via buying oversold dips in uptrend and bear markets via selling overbought rallies in downtrend
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_WilliamsR_Extreme_12hEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 12h data ONCE before loop (MTF Rule #1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*6h = 144h = 6 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24, period)  # warmup for EMA, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and price above/below EMA50 for trend alignment
            if curr_volume_spike:
                # Bullish entry: Williams %R crosses above -80 from below (leaving oversold) with price > EMA50_12h
                if (curr_williams_r > -80 and 
                    williams_r[i-1] <= -80 and 
                    curr_close > curr_ema_50_12h):
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R crosses below -20 from above (leaving overbought) with price < EMA50_12h
                elif (curr_williams_r < -20 and 
                      williams_r[i-1] >= -20 and 
                      curr_close < curr_ema_50_12h):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R reaches overbought (-20) or price crosses below EMA50_12h
            if curr_williams_r >= -20 or curr_close < curr_ema_50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R reaches oversold (-80) or price crosses above EMA50_12h
            if curr_williams_r <= -80 or curr_close > curr_ema_50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals