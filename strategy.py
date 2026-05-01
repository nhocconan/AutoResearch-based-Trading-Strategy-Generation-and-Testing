#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND 1w EMA50 is rising AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower band AND 1w EMA50 is falling AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 12h timeframe to capture medium-term trends with low trade frequency.
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) via 1w EMA50 slope filter.

name = "12h_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w EMA50 slope (rising/falling) using 3-bar difference
    ema_50_slope = np.zeros_like(ema_50_1w_aligned)
    ema_50_slope[3:] = ema_50_1w_aligned[3:] - ema_50_1w_aligned[:-3]
    
    # Donchian(20) channels on 12h data
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 12h timeframe
        hour = hours[i]
        
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_50_slope[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_slope = ema_50_slope[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 1.5)
        
        # Donchian breakout conditions
        bullish_breakout = curr_high > curr_upper  # price breaks above upper band
        bearish_breakout = curr_low < curr_lower   # price breaks below lower band
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish Donchian breakout AND 1w EMA50 rising AND volume confirmation
            if (bullish_breakout and 
                curr_ema_slope > 0 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish Donchian breakout AND 1w EMA50 falling AND volume confirmation
            elif (bearish_breakout and 
                  curr_ema_slope < 0 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price retouches middle of Donchian channel OR EMA50 slope turns negative
            middle_channel = (curr_upper + curr_lower) / 2
            if (curr_close < middle_channel or 
                curr_ema_slope < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price retouches middle of Donchian channel OR EMA50 slope turns positive
            middle_channel = (curr_upper + curr_lower) / 2
            if (curr_close > middle_channel or 
                curr_ema_slope > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals