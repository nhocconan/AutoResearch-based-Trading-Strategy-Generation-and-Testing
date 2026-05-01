#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 (1d) AND 1w EMA34 rising AND volume > 2x 20-bar average.
# Short when price breaks below Camarilla S3 (1d) AND 1w EMA34 falling AND volume > 2x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 12h timeframe to capture multi-day trends with low trade frequency.
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) via 1w EMA34 slope filter.

name = "12h_Camarilla_R3S3_1wEMA34_VolumeSpike_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Camarilla levels (R3, S3) from previous 1d bar
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Typical price for Camarilla calculation
    typical_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = close_1d + (range_1d * 1.1 / 4.0)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 4.0)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
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
    
    # Volume confirmation: current 12h volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 12h timeframe
        hour = hours[i]
        
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Camarilla breakout signals (using previous bar's levels to avoid look-ahead)
        breakout_up = curr_high > camarilla_r3_aligned[i-1]  # break above previous R3
        breakout_down = curr_low < camarilla_s3_aligned[i-1]  # break below previous S3
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Camarilla R3 AND 1w EMA34 rising AND volume confirmation
            if (breakout_up and 
                ema_34_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Camarilla S3 AND 1w EMA34 falling AND volume confirmation
            elif (breakout_down and 
                  ema_34_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Camarilla S3 (stoploss) OR 1w EMA34 falls (trend change)
            if (curr_low < camarilla_s3_aligned[i] or 
                ema_34_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Camarilla R3 (stoploss) OR 1w EMA34 rises (trend change)
            if (curr_high > camarilla_r3_aligned[i] or 
                ema_34_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals