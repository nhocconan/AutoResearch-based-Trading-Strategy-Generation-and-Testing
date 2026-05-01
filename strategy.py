#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d EMA50 trend and volume confirmation.
# Long when: Williams %R crosses above -80 from below AND 1d close > 1d EMA50 AND 4h volume > 1.8x 20-period average
# Short when: Williams %R crosses below -20 from above AND 1d close < 1d EMA50 AND 4h volume > 1.8x 20-period average
# Uses Williams %R from primary 4h data for mean reversion signals, 1d EMA50 for trend alignment, volume spike for conviction.
# Target: 20-50 trades/year on 4h. Discrete sizing 0.25 to minimize fee drag while capturing significant moves.
# Works in bull (reversals with trend) and bear (reversals with trend) by trading with aligned 1d trend.

name = "4h_WilliamsR_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
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
    
    # Load 4h data ONCE before loop for Williams %R calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h Williams %R (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 4h primary timeframe (no additional delay needed)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for 1d EMA50 (need 50+ for safety)
    
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
        curr_ema_50 = ema_50_1d_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.8x 20-period average
        # Calculate 4h volume MA on the fly using aligned 4h data
        vol_4h = df_4h['volume'].values
        vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
        curr_vol_ma = vol_ma_4h_aligned[i]
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)
        
        # 1d trend filter
        uptrend_1d = curr_close > curr_ema_50
        downtrend_1d = curr_close < curr_ema_50
        
        # Williams %R cross conditions
        williams_r_prev = williams_r_aligned[i-1] if i > 0 else -50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 from below AND 1d uptrend AND volume confirmation
            if (curr_williams_r > -80 and williams_r_prev <= -80 and 
                uptrend_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND 1d downtrend AND volume confirmation
            elif (curr_williams_r < -20 and williams_r_prev >= -20 and 
                  downtrend_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (reversal) OR 1d trend turns down
            if (curr_williams_r < -50 and williams_r_prev >= -50) or \
               not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (reversal) OR 1d trend turns up
            if (curr_williams_r > -50 and williams_r_prev <= -50) or \
               not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals