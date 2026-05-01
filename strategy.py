#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA21 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band with 1w EMA21 uptrend and volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower band with 1w EMA21 downtrend and volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 1d timeframe to avoid overtrading.
# Works in bull (buy breakouts) and bear (sell breakdowns) via trend filter.

name = "1d_Donchian20_1wEMA21_VolumeConfirm_v1"
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
        # Session filter: trade all sessions for 1d timeframe
        hour = hours[i]
        
        if np.isnan(ema_21_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_21_1w = ema_21_1w_aligned[i]
        
        # Calculate Donchian bands from previous 20 periods (1d bars)
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        # Donchian upper band: highest high of last 20 bars (excluding current)
        highest_high = np.max(high[i-20:i])
        # Donchian lower band: lowest low of last 20 bars (excluding current)
        lowest_low = np.min(low[i-20:i])
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 1.5)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band AND price > 1w EMA21 AND volume confirmation
            if (curr_close > highest_high and 
                curr_close > curr_ema_21_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND price < 1w EMA21 AND volume confirmation
            elif (curr_close < lowest_low and 
                  curr_close < curr_ema_21_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower band (reversal) OR price < 1w EMA21 (trend violation)
            if (curr_close < lowest_low or 
                curr_close < curr_ema_21_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band (reversal) OR price > 1w EMA21 (trend violation)
            if (curr_close > highest_high or 
                curr_close > curr_ema_21_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals