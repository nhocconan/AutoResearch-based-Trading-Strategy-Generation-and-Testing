#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Donchian breakout captures momentum; 1w EMA50 ensures alignment with major trend
# Volume spike (>2.0x 20-period EMA) confirms institutional participation
# Works in bull markets (buy breakouts above upper band) and bear markets (sell breakdowns below lower band)
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid Donchian
        # Skip if any value is NaN
        if np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels on 1d data (lookback 20 periods)
        lookback_start = max(0, i - 19)
        period_high = np.max(high[lookback_start:i+1])
        period_low = np.min(low[lookback_start:i+1])
        
        # Volume confirmation: 20-period EMA on volume
        vol_lookback_start = max(0, i - 19)
        if vol_lookback_start <= i:
            vol_slice = volume[vol_lookback_start:i+1]
            vol_ema_20 = np.mean(vol_slice)  # Simple average for volume confirmation
        else:
            vol_ema_20 = 0
        
        # Volume spike: current volume > 2.0 x 20-period average volume (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20) if vol_ema_20 > 0 else False
        
        if position == 0:
            # Long: price breaks above upper Donchian + price above 1w EMA50 + volume spike
            if close[i] > period_high and close[i] > ema_50_1w_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + price below 1w EMA50 + volume spike
            elif close[i] < period_low and close[i] < ema_50_1w_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian OR price below 1w EMA50
            if close[i] < period_low or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian OR price above 1w EMA50
            if close[i] > period_high or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals