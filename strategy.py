#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend + volume confirmation.
# Long when price breaks above Donchian upper (20-period high) AND 12h EMA50 rising AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower (20-period low) AND 12h EMA50 falling AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 4h timeframe to capture medium-term trends with low trade frequency.
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) via 12h EMA50 slope filter.

name = "4h_Donchian20_12hEMA50_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 calculation
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 12h EMA50 slope (rising/falling)
    ema_50_slope = np.diff(ema_50_aligned, prepend=ema_50_aligned[0])
    ema_50_rising = ema_50_slope > 0
    ema_50_falling = ema_50_slope < 0
    
    # Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 4h timeframe
        hour = hours[i]
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_upper[i-1]  # break above previous upper band
        breakout_down = curr_low < donchian_lower[i-1]  # break below previous lower band
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian upper AND 12h EMA50 rising AND volume confirmation
            if (breakout_up and 
                ema_50_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian lower AND 12h EMA50 falling AND volume confirmation
            elif (breakout_down and 
                  ema_50_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian lower (stoploss) OR 12h EMA50 falls (trend change)
            if (curr_low < donchian_lower[i] or 
                ema_50_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper (stoploss) OR 12h EMA50 rises (trend change)
            if (curr_high > donchian_upper[i] or 
                ema_50_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals