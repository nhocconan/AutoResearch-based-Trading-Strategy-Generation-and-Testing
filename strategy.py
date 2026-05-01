#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA21 trend filter and volume spike confirmation.
# Long when price breaks above upper Donchian with 1w EMA21 uptrend and volume > 2.0x 20-bar average.
# Short when price breaks below lower Donchian with 1w EMA21 downtrend and volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 1d timeframe to avoid overtrading.
# Works in bull (buy breakouts) and bear (sell breakdowns) via trend filter.

name = "1d_Donchian20_1wEMA21_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop for EMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # 1w EMA21 for trend filter
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for EMA21 and Donchian calculation
    
    for i in range(start_idx, n):
        # Session filter: 00-23 UTC (trade all sessions for 1d timeframe)
        hour = hours[i]
        
        if np.isnan(ema_21_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_21_1w = ema_21_1w_aligned[i]
        
        # Calculate Donchian levels from previous 20 1d bars
        # Use bars i-20 to i-1 for prior period calculation (20 bars lookback)
        if i-20 < start_idx:
            signals[i] = 0.0
            continue
            
        lookback_high = np.max(high[i-20:i])  # highest high of last 20 bars
        lookback_low = np.min(low[i-20:i])    # lowest low of last 20 bars
        
        if lookback_high <= lookback_low:
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current 1d volume > 2.0x 20-period average
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
            # Long: price breaks above upper Donchian AND price > 1w EMA21 AND volume confirmation
            if (curr_close > lookback_high and 
                curr_close > curr_ema_21_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND price < 1w EMA21 AND volume confirmation
            elif (curr_close < lookback_low and 
                  curr_close < curr_ema_21_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower Donchian (reversal) OR price < 1w EMA21 (trend violation)
            if (curr_close < lookback_low or 
                curr_close < curr_ema_21_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian (reversal) OR price > 1w EMA21 (trend violation)
            if (curr_close > lookback_high or 
                curr_close > curr_ema_21_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals