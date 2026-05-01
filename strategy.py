#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA50 trend filter and volume confirmation.
# Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100 over 14 periods.
# Long: %R < -80 (oversold) AND price > 1d EMA50 AND volume > 1.5x 20-bar average.
# Short: %R > -20 (overbought) AND price < 1d EMA50 AND volume > 1.5x 20-bar average.
# Williams %R identifies exhaustion points; 1d EMA50 ensures alignment with higher timeframe trend.
# Volume confirmation filters weak signals. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Target: 12-37 trades/year on 6h (50-150 total over 4 years). Discrete sizing 0.25 to minimize fee drag.

name = "6h_WilliamsR_1dEMA50_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 6h data ONCE before loop for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(df_6h['high'].values).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_6h['low'].values).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - df_6h['close'].values) / (highest_high - lowest_low) * -100
    williams_r = williams_r.values  # Convert to numpy array
    
    # Align Williams %R to 6h primary timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h primary timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R (14) + EMA50 (50)
    
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
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        vol_6h = df_6h['volume'].values
        vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
        vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
        curr_vol_ma = vol_ma_6h_aligned[i]
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume confirmation
            if (curr_williams_r < -80 and 
                curr_close > curr_ema_50_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume confirmation
            elif (curr_williams_r > -20 and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR price < 1d EMA50 (trend violation)
            if (curr_williams_r > -20 or 
                curr_close < curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR price > 1d EMA50 (trend violation)
            if (curr_williams_r < -80 or 
                curr_close > curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals