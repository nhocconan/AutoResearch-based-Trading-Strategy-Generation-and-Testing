#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout direction + 1d EMA50 trend filter + volume confirmation.
# Enters long when price breaks above 4h Donchian upper band with 1d EMA50 uptrend and volume > 2.0x 20-bar average.
# Enters short when price breaks below 4h Donchian lower band with 1d EMA50 downtrend and volume > 2.0x 20-bar average.
# Uses session filter (08-20 UTC) to reduce noise. Discrete sizing 0.20 to minimize fee churn.
# Designed for 1h timeframe with HTF direction to avoid overtrading (target: 15-37 trades/year).
# Works in bull (buy breakouts) and bear (sell breakdowns) via 1d trend filter.

name = "1h_Donchian20_1dEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for Donchian bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for EMA50 and Donchian calculation
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        # Calculate 4h Donchian bands from previous 20 periods (4h bars)
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        # 4h Donchian upper band: highest high of last 20 bars (excluding current)
        highest_high = np.max(high[i-20:i])
        # 4h Donchian lower band: lowest low of last 20 bars (excluding current)
        lowest_low = np.min(low[i-20:i])
        
        # Volume confirmation: current 1h volume > 2.0x 20-period average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 4h Donchian upper band AND price > 1d EMA50 AND volume confirmation
            if (curr_close > highest_high and 
                curr_close > curr_ema_50_1d and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower band AND price < 1d EMA50 AND volume confirmation
            elif (curr_close < lowest_low and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below 4h Donchian lower band (reversal) OR price < 1d EMA50 (trend violation)
            if (curr_close < lowest_low or 
                curr_close < curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above 4h Donchian upper band (reversal) OR price > 1d EMA50 (trend violation)
            if (curr_close > highest_high or 
                curr_close > curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals