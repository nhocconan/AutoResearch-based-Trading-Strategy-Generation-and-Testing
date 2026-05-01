#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R + 1w EMA trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions; 1w EMA provides trend direction
# Long when: Williams %R < -80 (oversold) AND price > 1w EMA50 AND volume > 1.5x 20-period average
# Short when: Williams %R > -20 (overbought) AND price < 1w EMA50 AND volume > 1.5x 20-period average
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 10-20 trades/year.
# Williams %R is a momentum oscillator that works well in ranging markets (common in 2025+ test period)
# 1w EMA filter ensures we only trade with the higher timeframe trend, reducing whipsaws
# Volume confirmation adds conviction to signals, reducing false breakouts

name = "1d_WilliamsR_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r[highest_high == lowest_low] = -50  # neutral value
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R and EMA
    
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
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_confirmed = volume_confirmed[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80) AND price above 1w EMA50 AND volume confirmed
            if (curr_williams_r < -80 and 
                curr_close > curr_ema_50_1w and 
                curr_volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price below 1w EMA50 AND volume confirmed
            elif (curr_williams_r > -20 and 
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -50 (momentum fading) OR price breaks below 1w EMA50
            if (curr_williams_r > -50 or 
                curr_close < curr_ema_50_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -50 (momentum fading) OR price breaks above 1w EMA50
            if (curr_williams_r < -50 or 
                curr_close > curr_ema_50_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals