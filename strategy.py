#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above R3 with 1d EMA34 uptrend and volume > 2x 20-bar average.
# Short when price breaks below S3 with 1d EMA34 downtrend and volume > 2x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 12h timeframe to capture medium-term swings.
# Works in bull (buy R3 breakout in uptrend) and bear (sell S3 breakout in downtrend) via EMA filter.

name = "12h_Camarilla_R3_S3_1dEMA34_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 calculation
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for each 12h bar using prior 1d OHLC
    # Need to align prior day's OHLC to current 12h bar
    # For 12h bar at time t, use 1d bar that closed at or before t-12h
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d OHLC
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    rng = high_1d - low_1d
    r3 = close_1d + 1.1 * rng
    s3 = close_1d - 1.1 * rng
    
    # Align Camarilla levels to 12h timeframe (use prior day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for EMA and volume
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 12h timeframe
        hour = hours[i]
        
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_open = open_1d[i] if i < len(open_1d) else open_1d[-1]  # not used but for safety
        
        # Volume confirmation: current 12h volume > 2x 20-period average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 2.0)
        
        # EMA trend: rising if current > prior, falling if current < prior
        if i > 0:
            ema_prior = ema_34_1d_aligned[i-1]
            ema_rising = curr_ema > ema_prior
            ema_falling = curr_ema < ema_prior
        else:
            ema_rising = False
            ema_falling = False
        
        # Breakout conditions
        # Long: price breaks above R3 AND EMA rising AND volume confirmation
        # Short: price breaks below S3 AND EMA falling AND volume confirmation
        long_breakout = curr_high > curr_r3
        short_breakout = curr_low < curr_s3
        
        # Entry logic
        if position == 0:  # Flat - look for new entries
            if (long_breakout and 
                ema_rising and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            elif (short_breakout and 
                  ema_falling and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3 (reversal) OR EMA falls (trend change)
            if (curr_low < curr_s3 or 
                not ema_rising):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (reversal) OR EMA rises (trend change)
            if (curr_high > curr_r3 or 
                not ema_falling):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals