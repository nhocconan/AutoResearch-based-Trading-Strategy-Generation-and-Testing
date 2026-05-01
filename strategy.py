#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper channel (20-period high) with 1w EMA50 uptrend and volume > 2x 20-bar average.
# Short when price breaks below Donchian lower channel (20-period low) with 1w EMA50 downtrend and volume > 2x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 1d timeframe to capture medium-term trends with low trade frequency.
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) via 1w EMA50 filter.

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
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
    
    # 1w EMA50 calculation
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channels on 1d data (20-period)
    # Upper channel: 20-period high
    # Lower channel: 20-period low
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 1d timeframe
        hour = hours[i]
        
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema = ema_50_1w_aligned[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        
        # Volume confirmation: current 1d volume > 2x 20-period average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 2.0)
        
        # Donchian breakout signals
        # Long breakout: price closes above upper channel
        # Short breakdown: price closes below lower channel
        long_breakout = curr_close > curr_upper
        short_breakout = curr_close < curr_lower
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish breakout AND 1w EMA50 uptrend (price > EMA) AND volume confirmation
            if (long_breakout and 
                curr_close > curr_ema and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish breakdown AND 1w EMA50 downtrend (price < EMA) AND volume confirmation
            elif (short_breakout and 
                  curr_close < curr_ema and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below lower channel (breakdown) OR 1w EMA50 turns down (price < EMA)
            if (curr_close < curr_lower or 
                curr_close < curr_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above upper channel (breakout) OR 1w EMA50 turns up (price > EMA)
            if (curr_close > curr_upper or 
                curr_close > curr_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals