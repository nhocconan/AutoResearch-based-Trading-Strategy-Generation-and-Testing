#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above upper Donchian with 12h EMA50 uptrend and volume > 1.8x 20-bar average.
# Short when price breaks below lower Donchian with 12h EMA50 downtrend and volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 4h timeframe to avoid overtrading.
# Works in bull (buy breakouts) and bear (sell breakdowns) via trend filter.

name = "4h_Donchian20_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for EMA50 and Donchian calculation
    
    for i in range(start_idx, n):
        # Session filter: 00-23 UTC (trade all sessions for 4h timeframe)
        hour = hours[i]
        
        if np.isnan(ema_50_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        
        # Calculate Donchian levels from previous 20 4h bars
        # Use bars i-20 to i-1 for prior period calculation (20 bars lookback)
        if i-20 < start_idx:
            signals[i] = 0.0
            continue
            
        lookback_high = np.max(high[i-20:i])  # highest high of last 20 bars
        lookback_low = np.min(low[i-20:i])    # lowest low of last 20 bars
        
        if lookback_high <= lookback_low:
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current 4h volume > 1.8x 20-period average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 1.8)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian AND price > 12h EMA50 AND volume confirmation
            if (curr_close > lookback_high and 
                curr_close > curr_ema_50_12h and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND price < 12h EMA50 AND volume confirmation
            elif (curr_close < lookback_low and 
                  curr_close < curr_ema_50_12h and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower Donchian (reversal) OR price < 12h EMA50 (trend violation)
            if (curr_close < lookback_low or 
                curr_close < curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian (reversal) OR price > 12h EMA50 (trend violation)
            if (curr_close > lookback_high or 
                curr_close > curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals