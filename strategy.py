#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 12h EMA50 trend filter and volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND price > 12h EMA50 AND volume > 1.5x 20-bar average.
# Short when Williams %R > -20 (overbought) AND price < 12h EMA50 AND volume > 1.5x 20-bar average.
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
# Williams %R identifies extreme reversals, 12h EMA50 ensures trend alignment, volume spike confirms conviction.
# Target: 30-60 trades/year on 4h. Discrete sizing 0.25 to minimize fee drag while capturing mean reversion moves.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) by trading with aligned 12h trend.

name = "4h_WilliamsR_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for Williams %R calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_4h['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_4h['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_4h['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align Williams %R to 4h primary timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Calculate 4h EMA20 for dynamic exit (optional, using Williams %R crossover instead)
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h primary timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 55  # warmup for Williams %R (14) + EMA50 (50) + EMA20 (20)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_ema_20_4h = ema_20_4h_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        vol_4h = df_4h['volume'].values
        vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
        curr_vol_ma = vol_ma_4h_aligned[i]
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) + price > 12h EMA50 + volume confirmation
            if (curr_williams_r < -80 and 
                curr_close > curr_ema_50_12h and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + price < 12h EMA50 + volume confirmation
            elif (curr_williams_r > -20 and 
                  curr_close < curr_ema_50_12h and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (mean reversion complete)
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (mean reversion complete)
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals