#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1w trend filter and volume confirmation.
# Long when: price breaks above Donchian upper (20) AND 1w close > 1w EMA50 AND 4h volume > 2.0x 20-period average
# Short when: price breaks below Donchian lower (20) AND 1w close < 1w EMA50 AND 4h volume > 2.0x 20-period average
# Uses Donchian channels for structure, 1w EMA50 for trend alignment, volume spike for conviction.
# Target: 15-35 trades/year on 4h. Discrete sizing 0.25 to minimize fee drag while capturing trends.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) by trading with aligned 1w trend.

name = "4h_Donchian20_1wEMA50_VolumeConfirm_v1"
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
    
    # Load 4h data ONCE before loop for price action and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for Donchian calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper (20-period high)
    donch_hi_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian lower (20-period low)
    donch_lo_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h primary timeframe
    donch_hi_aligned = align_htf_to_ltf(prices, df_1w, donch_hi_20)
    donch_lo_aligned = align_htf_to_ltf(prices, df_1w, donch_lo_20)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 4h volume average (20-period) for volume confirmation
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for 1w EMA50
    
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
        if (np.isnan(donch_hi_aligned[i]) or np.isnan(donch_lo_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_4h_aligned[i]
        curr_donch_hi = donch_hi_aligned[i]
        curr_donch_lo = donch_lo_aligned[i]
        curr_ema_50 = ema_50_aligned[i]
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # 1w trend filter
        uptrend_1w = curr_close > curr_ema_50
        downtrend_1w = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian upper AND 1w uptrend AND volume confirmation
            if (curr_high > curr_donch_hi and 
                uptrend_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower AND 1w downtrend AND volume confirmation
            elif (curr_low < curr_donch_lo and 
                  downtrend_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below Donchian upper (breakdown) OR 1w trend turns down
            if (curr_close < curr_donch_hi or 
                not uptrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian lower (breakout) OR 1w trend turns up
            if (curr_close > curr_donch_lo or 
                not downtrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals