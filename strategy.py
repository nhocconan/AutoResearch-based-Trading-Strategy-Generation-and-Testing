#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 AND 1w EMA34 rising AND volume > 1.8x 20-bar average.
# Short when price breaks below S3 AND 1w EMA34 falling AND volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 12h timeframe to capture swing trends.
# Camarilla levels provide adaptive support/resistance based on prior day's range.
# 1w EMA34 trend filter ensures alignment with higher timeframe momentum.
# Volume spike requirement reduces false breakouts and improves signal quality.
# Target: 50-150 total trades over 4 years (12-37/year) for BTC/ETH/SOL.

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 calculation
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # 1w EMA34 slope (rising/falling)
    ema_34_slope = np.diff(ema_34_aligned, prepend=ema_34_aligned[0])
    ema_34_rising = ema_34_slope > 0
    ema_34_falling = ema_34_slope < 0
    
    # Calculate Camarilla levels (R3, S3) from prior 1d bar
    # Need to shift by 1 to avoid look-ahead: use prior completed 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d bar's high, low, close (shifted by 1 bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get prior completed 1d bar values
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # Set first value to NaN (no prior bar)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels for prior 1d bar
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close_1d_prev + (high_1d_prev - low_1d_prev) * 1.1 / 4
    camarilla_s3 = close_1d_prev - (high_1d_prev - low_1d_prev) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current 12h volume > 1.8x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and Donchian calculation
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 12h timeframe
        hour = hours[i]
        
        if np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3_aligned[i]  # break above R3
        breakout_down = curr_low < camarilla_s3_aligned[i]   # break below S3
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND 1w EMA34 rising AND volume confirmation
            if (breakout_up and 
                ema_34_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 AND 1w EMA34 falling AND volume confirmation
            elif (breakout_down and 
                  ema_34_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR 1w EMA34 falls (trend change)
            if (curr_low < camarilla_s3_aligned[i] or 
                ema_34_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR 1w EMA34 rises (trend change)
            if (curr_high > camarilla_r3_aligned[i] or 
                ema_34_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals