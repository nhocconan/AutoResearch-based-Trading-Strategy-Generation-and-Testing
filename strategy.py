#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike confirmation.
# Long when price breaks above Donchian upper channel AND 12h EMA50 is rising AND volume > 2x 20-bar average.
# Short when price breaks below Donchian lower channel AND 12h EMA50 is falling AND volume > 2x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 4h timeframe with proven BTC/ETH edge.
# Works in bull (buy breakouts in uptrend) and bear (sell breakouts in downtrend) via 12h EMA50 slope filter.

name = "4h_Donchian20_12hEMA50_VolumeSpike_v1"
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
    ema_50 = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_50_slope = np.gradient(ema_50_aligned)  # positive = rising, negative = falling
    
    # Donchian channels on 4h data (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for Donchian and EMA50
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 4h timeframe
        hour = hours[i]
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_50_slope[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol_spike = volume_spike[i]
        curr_ema_slope = ema_50_slope[i]
        
        # Breakout conditions
        bullish_breakout = curr_high > donchian_upper[i-1]  # break above previous upper channel
        bearish_breakout = curr_low < donchian_lower[i-1]   # break below previous lower channel
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish breakout AND rising 12h EMA50 AND volume spike
            if (bullish_breakout and 
                curr_ema_slope > 0 and 
                curr_vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout AND falling 12h EMA50 AND volume spike
            elif (bearish_breakout and 
                  curr_ema_slope < 0 and 
                  curr_vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price re-enters Donchian channel OR 12h EMA50 slope turns negative
            if (curr_close < donchian_upper[i] or 
                curr_ema_slope < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price re-enters Donchian channel OR 12h EMA50 slope turns positive
            if (curr_close > donchian_lower[i] or 
                curr_ema_slope > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals