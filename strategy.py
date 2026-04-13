#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h timeframe with 4h/1d HTF filters
    # Use 4h trend direction (EMA21) + 1d volume regime (high/low) + 1h entry timing (pullback to EMA8)
    # Long: 4h EMA21 up + 1d volume > 20-day avg + 1h price pulls back to EMA8 from above
    # Short: 4h EMA21 down + 1d volume < 20-day avg + 1h price bounces off EMA8 from below
    # Session filter: 08-20 UTC only
    # Position size: 0.20 (20%) to control drawdown
    # Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend direction (EMA21)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Get 1d data for volume regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Get 1h EMA8 for entry timing
    ema_8 = pd.Series(close).ewm(span=8, min_periods=8, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(30, n):
        # Skip if not in session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(ema_21_4h_aligned[i-1]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(ema_8[i])):
            signals[i] = 0.0
            continue
        
        # 4h trend direction: EMA21 slope
        ema_21_4h_up = ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1]
        ema_21_4h_down = ema_21_4h_aligned[i] < ema_21_4h_aligned[i-1]
        
        # 1d volume regime: above/below average
        volume_high = volume_1d[i] > vol_avg_20_1d_aligned[i]
        volume_low = volume_1d[i] < vol_avg_20_1d_aligned[i]
        
        # 1h entry timing: price relative to EMA8
        price_above_ema8 = close[i] > ema_8[i]
        price_below_ema8 = close[i] < ema_8[i]
        was_above_ema8 = close[i-1] > ema_8[i-1]
        was_below_ema8 = close[i-1] < ema_8[i-1]
        
        # Entry conditions
        enter_long = (ema_21_4h_up and volume_high and 
                     price_above_ema8 and was_below_ema8)
        enter_short = (ema_21_4h_down and volume_low and 
                      price_below_ema8 and was_above_ema8)
        
        # Exit conditions: trend reversal or volume regime change
        exit_long = (position == 1 and 
                    (ema_21_4h_down or not volume_high))
        exit_short = (position == -1 and 
                     (ema_21_4h_up or not volume_low))
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_ema_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0