#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when: price breaks above Donchian(20) high AND 1d close > 1d EMA50 AND 4h volume > 2x 20-period average
# Short when: price breaks below Donchian(20) low AND 1d close < 1d EMA50 AND 4h volume > 2x 20-period average
# Uses discrete sizing 0.25. Target: 20-50 trades/year on 4h.
# Donchian provides clear price structure, 1d EMA50 ensures higher timeframe alignment, volume spike confirms conviction.
# Works in bull (catching breakouts) and bear (catching breakdowns) by trading with the 1d trend.

name = "4h_Donchian20_1dEMA50_VolumeSpike_v1"
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
    
    # Load 4h data ONCE before loop for Donchian and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian(20) on 4h
    donch_high_20 = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h primary timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h volume average (20-period) for volume spike confirmation
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
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_4h_aligned[i]
        curr_donch_high = donch_high_aligned[i]
        curr_donch_low = donch_low_aligned[i]
        curr_ema_50 = ema_50_aligned[i]
        
        # Volume spike: current 4h volume > 2x 20-period average
        volume_spike = curr_vol > (curr_vol_ma * 2.0)
        
        # Breakout conditions
        bullish_breakout = curr_close > curr_donch_high
        bearish_breakout = curr_close < curr_donch_low
        
        # 1d trend filter: price above/below EMA50
        uptrend_1d = curr_close > curr_ema_50
        downtrend_1d = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish breakout AND 1d uptrend AND volume spike
            if (bullish_breakout and 
                uptrend_1d and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout AND 1d downtrend AND volume spike
            elif (bearish_breakout and 
                  downtrend_1d and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below Donchian low OR 1d trend turns down
            if (curr_close < curr_donch_low or 
                not uptrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR 1d trend turns up
            if (curr_close > curr_donch_high or 
                not downtrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals