#!/usr/bin/env python3
"""
12h_1w_1d_donchian_breakout_volume_v1
Hypothesis: 12h Donchian channel breakout with 1w trend filter and 1d volume confirmation.
- Long: 12h close breaks above Donchian(20) high + 1w close > EMA200 (1w) + 1d volume > 1.5x 20-day avg
- Short: 12h close breaks below Donchian(20) low + 1w close < EMA200 (1w) + 1d volume > 1.5x 20-day avg
- Exit: Opposite Donchian breakout or trend reversal
- Position sizing: 0.25 long, -0.25 short
- Designed to capture trends with confirmation, avoiding false breakouts in ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # 1w EMA(200) for trend
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_1w_up = close_1w > ema_200_1w
    trend_1w_down = close_1w < ema_200_1w
    
    # Forward fill trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d volume MA to 12h
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Donchian channels on 12h
    # We need 12h high/low, so get 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian(20) - highest high and lowest low of last 20 periods
    high_max_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (they are already 12h, but need to align to 12h index of prices)
    high_max_20_aligned = align_htf_to_ltf(prices, df_12h, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_12h, low_min_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup - need enough for Donchian calculation
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(high_max_20_aligned[i]) or 
            np.isnan(low_min_20_aligned[i]) or np.isnan(volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian low OR 1w trend turns down
            if (close[i] < low_min_20_aligned[i]) or trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR 1w trend turns up
            if (close[i] > high_max_20_aligned[i]) or trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Close breaks above Donchian high + 1w uptrend + volume confirmation
            if (close[i] > high_max_20_aligned[i]) and trend_1w_up_aligned[i] and (volume[i] > 1.5 * vol_ma_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Close breaks below Donchian low + 1w downtrend + volume confirmation
            elif (close[i] < low_min_20_aligned[i]) and trend_1w_down_aligned[i] and (volume[i] > 1.5 * vol_ma_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals