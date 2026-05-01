#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 12h EMA trend filter.
# Long when: price breaks above Donchian(20) high AND 1d volume > 2.0x 20-period average AND price > 12h EMA50
# Short when: price breaks below Donchian(20) low AND 1d volume > 2.0x 20-period average AND price < 12h EMA50
# Uses discrete sizing 0.25 to control fees. Target: 30-60 trades/year.
# Donchian provides objective breakout levels, volume spike confirms conviction, 12h EMA filters counter-trend noise.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) by trading with intermediate trend.

name = "4h_Donchian20_1dVolumeSpike_12hEMA50_Trend_v1"
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
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
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
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol_ma = vol_ma_1d_aligned[i]
        curr_ema = ema_50_12h_aligned[i]
        curr_donch_high = highest_high_20[i]
        curr_donch_low = lowest_low_20[i]
        
        # Get current 1d volume (need to align to 4h)
        # 1d = 6 * 4h bars
        idx_1d = i // 6
        if idx_1d < len(df_1d):
            curr_vol_1d = df_1d['volume'].iloc[idx_1d]
        else:
            curr_vol_1d = 0
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        volume_confirmed = curr_vol_1d > (curr_vol_ma * 2.0)
        
        # Trend filter: price above/below 12h EMA50
        uptrend = curr_close > curr_ema
        downtrend = curr_close < curr_ema
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian high AND volume confirmed AND uptrend
            if (curr_close > curr_donch_high and 
                volume_confirmed and 
                uptrend):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND volume confirmed AND downtrend
            elif (curr_close < curr_donch_low and 
                  volume_confirmed and 
                  downtrend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR price crosses below 12h EMA50
            if (curr_close < curr_donch_low or 
                curr_close < curr_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR price crosses above 12h EMA50
            if (curr_close > curr_donch_high or 
                curr_close > curr_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals