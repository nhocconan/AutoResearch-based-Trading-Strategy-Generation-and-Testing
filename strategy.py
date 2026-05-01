#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# Long when: price breaks above 20-day high AND 1w close > 1w EMA50 AND 1d volume > 1.5x 20-day average
# Short when: price breaks below 20-day low AND 1w close < 1w EMA50 AND 1d volume > 1.5x 20-day average
# Uses discrete sizing 0.25. Target: 15-30 trades/year on 1d.
# Donchian captures breakouts, 1w EMA50 filters for higher timeframe trend, volume confirms conviction.
# Works in bull (catching strong uptrends) and bear (catching breakdowns) by trading with the 1w trend.

name = "1d_Donchian20_1wTrend_VolumeConfirm_v1"
timeframe = "1d"
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
    
    # Load 1d data ONCE before loop for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d Donchian channels: 20-period high/low
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (already aligned, but use helper for consistency)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d volume average (20-period) for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
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
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_high_20 = high_20_aligned[i]
        curr_low_20 = low_20_aligned[i]
        curr_ema_50 = ema_50_aligned[i]
        curr_vol_ma = vol_ma_1d_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # 1w trend filter: price above/below EMA50
        uptrend_1w = curr_close > curr_ema_50
        downtrend_1w = curr_close < curr_ema_50
        
        # Breakout conditions
        breakout_up = curr_high > curr_high_20  # price breaks above 20-day high
        breakout_down = curr_low < curr_low_20  # price breaks below 20-day low
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish breakout AND 1w uptrend AND volume confirmation
            if (breakout_up and 
                uptrend_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout AND 1w downtrend AND volume confirmation
            elif (breakout_down and 
                  downtrend_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below 20-day low OR 1w trend turns down
            if (curr_close < curr_low_20 or 
                not uptrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above 20-day high OR 1w trend turns up
            if (curr_close > curr_high_20 or 
                not downtrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals