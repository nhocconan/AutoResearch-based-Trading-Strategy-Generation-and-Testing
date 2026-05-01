#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d EMA34 trend + volume confirmation.
# Long when: Williams %R(14) crosses above -80 (oversold) AND price > 1d EMA34 AND 6h volume > 1.5x 20-period average
# Short when: Williams %R(14) crosses below -20 (overbought) AND price < 1d EMA34 AND 6h volume > 1.5x 20-period average
# Williams %R identifies reversal points in ranging markets; EMA34 filters for major trend alignment; volume confirms conviction.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) by trading with aligned 1d trend.
# Target: 12-30 trades/year on 6h. Discrete sizing 0.25 to minimize fee drag while capturing mean-reversion moves.

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for Williams %R and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h primary timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load 6h data ONCE before loop for volume average
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for 1d EMA34 (need 34+ for safety)
    
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
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_6h_aligned[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema_34 = ema_34_1d_aligned[i]
        
        # Previous bar values for crossover detection
        prev_williams_r = williams_r_aligned[i-1] if i > 0 else -50
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # 1d trend filter
        uptrend_1d = curr_close > curr_ema_34
        downtrend_1d = curr_close < curr_ema_34
        
        # Williams %R crossover conditions
        williams_r_cross_above_80 = (prev_williams_r <= -80) and (curr_williams_r > -80)
        williams_r_cross_below_20 = (prev_williams_r >= -20) and (curr_williams_r < -20)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 (oversold) AND 1d uptrend AND volume confirmation
            if (williams_r_cross_above_80 and 
                uptrend_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) AND 1d downtrend AND volume confirmation
            elif (williams_r_cross_below_20 and 
                  downtrend_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR 1d trend turns down
            if (williams_r_cross_below_20 or 
                not uptrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR 1d trend turns up
            if (williams_r_cross_above_80 or 
                not downtrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals