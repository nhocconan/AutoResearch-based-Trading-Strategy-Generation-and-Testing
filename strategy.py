#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when: price breaks above 20-bar Donchian high AND 12h close > 12h EMA50 AND 6h volume > 1.5x 20-period average
# Short when: price breaks below 20-bar Donchian low AND 12h close < 12h EMA50 AND 6h volume > 1.5x 20-period average
# Uses discrete sizing 0.25. Target: 12-37 trades/year on 6h.
# Donchian provides structure, 12h EMA50 filters for higher timeframe trend alignment, volume spike confirms conviction.
# Works in bull (catching breakouts with trend) and bear (catching breakdowns with trend) by trading with the 12h trend.

name = "6h_Donchian20_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Load 6h data ONCE before loop for Donchian channels and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 6h Donchian(20) channels
    donch_high_6h = pd.Series(df_6h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low_6h = pd.Series(df_6h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h primary timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_6h, donch_high_6h)
    donch_low_aligned = align_htf_to_ltf(prices, df_6h, donch_low_6h)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 6h volume average (20-period) for volume spike confirmation
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_6h_aligned[i]
        curr_donch_high = donch_high_aligned[i]
        curr_donch_low = donch_low_aligned[i]
        curr_ema_50 = ema_50_aligned[i]
        
        # Volume spike: current 6h volume > 1.5x 20-period average
        volume_spike = curr_vol > (curr_vol_ma * 1.5)
        
        # Breakout conditions
        breakout_up = curr_close > curr_donch_high
        breakout_down = curr_close < curr_donch_low
        
        # 12h trend filter
        uptrend_12h = curr_close > curr_ema_50
        downtrend_12h = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish breakout AND 12h uptrend AND volume spike
            if (breakout_up and 
                uptrend_12h and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout AND 12h downtrend AND volume spike
            elif (breakout_down and 
                  downtrend_12h and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below 12h EMA50 (trend change) OR Donchian breakdown
            if (curr_close < curr_ema_50 or 
                curr_close < curr_donch_low):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above 12h EMA50 (trend change) OR Donchian breakout
            if (curr_close > curr_ema_50 or 
                curr_close > curr_donch_high):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals