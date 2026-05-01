#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when: price breaks above Donchian(20) high AND 12h EMA50 rising AND volume > 1.5x 20-period average
# Short when: price breaks below Donchian(20) low AND 12h EMA50 falling AND volume > 1.5x 20-period average
# Uses discrete sizing 0.25. Target: 20-50 trades/year on 4h.
# Donchian captures breakouts, 12h EMA50 filters for higher timeframe trend alignment, volume confirms conviction.
# Works in bull (catching uptrend breakouts) and bear (catching downtrend breakdowns) by trading with the aligned trend.

name = "4h_Donchian20_12hEMA50_VolumeConfirm_v1"
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
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Donchian(20) on 4h: 20-period high and low
    donchian_high = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h primary timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h volume average (20-period) for volume confirmation
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
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
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_4h_aligned[i]
        curr_donchian_high = donchian_high_aligned[i]
        curr_donchian_low = donchian_low_aligned[i]
        curr_ema_50 = ema_50_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # 12h EMA50 trend: rising if current > previous, falling if current < previous
        if i > start_idx:
            prev_ema_50 = ema_50_aligned[i-1]
            ema_rising = curr_ema_50 > prev_ema_50
            ema_falling = curr_ema_50 < prev_ema_50
        else:
            ema_rising = False
            ema_falling = False
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian high AND EMA50 rising AND volume confirmation
            if (curr_close > curr_donchian_high and 
                ema_rising and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND EMA50 falling AND volume confirmation
            elif (curr_close < curr_donchian_low and 
                  ema_falling and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below Donchian low (breakdown) OR EMA50 starts falling
            if (curr_close < curr_donchian_low or 
                ema_falling):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high (breakout) OR EMA50 starts rising
            if (curr_close > curr_donchian_high or 
                ema_rising):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals