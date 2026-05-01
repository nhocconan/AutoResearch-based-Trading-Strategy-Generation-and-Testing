#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian channel breakout with 4h trend filter and volume confirmation.
# Uses 4h EMA50 for trend direction and 1d volume spike (2x 20-period average) for confirmation.
# Long when: price breaks above 1h Donchian(20) high AND 4h EMA50 up AND 1d volume > 2x 20-period average.
# Short when: price breaks below 1h Donchian(20) low AND 4h EMA50 down AND 1d volume > 2x 20-period average.
# Uses discrete sizing 0.20 to minimize fee churn. Target: 15-35 trades/year (~60-140 over 4 years).
# Donchian provides objective breakout levels, 4h EMA filters counter-trend trades, volume confirms conviction.
# Works in bull (follows 4h trend) and bear (avoids false breakouts via volume + trend filter).

name = "1h_Donchian20_4hEMA50_1dVolumeSpike_v1"
timeframe = "1h"
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
    
    # Load 4h data ONCE before loop for EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1h Donchian channel (20-period)
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donchian_high = high_roll_max[i]
        curr_donchian_low = low_roll_min[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_vol_ma_1d = vol_ma_20_1d_aligned[i]
        
        # Volume spike condition: current 1h volume > 2x 1d volume MA (scaled)
        # Approximate: 1h volume > (1d volume MA / 24) * 2 = 1d volume MA / 12
        vol_spike = curr_volume > (curr_vol_ma_1d / 12.0) * 2.0
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian high AND 4h EMA50 up AND volume spike
            if (curr_close > curr_donchian_high and 
                curr_close > curr_ema_4h and 
                vol_spike):
                signals[i] = 0.20
                position = 1
            # Short: break below Donchian low AND 4h EMA50 down AND volume spike
            elif (curr_close < curr_donchian_low and 
                  curr_close < curr_ema_4h and 
                  vol_spike):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR 4h EMA50 turns down
            if (curr_close < curr_donchian_low or 
                curr_close < curr_ema_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR 4h EMA50 turns up
            if (curr_close > curr_donchian_high or 
                curr_close > curr_ema_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals