#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band with 1d EMA50 uptrend and volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower band with 1d EMA50 downtrend and volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 4h timeframe to target 20-50 trades/year.
# Works in bull (buy breakouts) and bear (sell breakdowns) via trend filter.

name = "4h_Donchian20_1dEMA50_VolumeConfirm_v1"
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
        # Session filter: 00-23 UTC (trade all sessions for 4h timeframe)
        hour = hours[i]
        
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        # Calculate Donchian channels from previous 20 periods (lookback 20)
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        # Donchian upper/lower bands using previous 20 periods
        lookback_high = np.max(high[i-20:i])
        lookback_low = np.min(low[i-20:i])
        
        if lookback_high <= lookback_low:
            signals[i] = 0.0
            continue
            
        upper_band = lookback_high
        lower_band = lookback_low
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
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
            # Long: price breaks above upper band AND price > 1d EMA50 AND volume confirmation
            if (curr_close > upper_band and 
                curr_close > curr_ema_50_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND price < 1d EMA50 AND volume confirmation
            elif (curr_close < lower_band and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower band (reversal) OR price < 1d EMA50 (trend violation)
            if (curr_close < lower_band or 
                curr_close < curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper band (reversal) OR price > 1d EMA50 (trend violation)
            if (curr_close > upper_band or 
                curr_close > curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals