#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakouts for direction and 1h volume spike for entry timing.
# 4h Donchian provides structural trend filter (proven edge), reducing false breakouts.
# Entry only on 1h bar when volume > 1.5x 20-period EMA and price breaks 4h Donchian level.
# Stop loss: exit when price closes below/above opposite Donchian level.
# Session filter: 08-20 UTC to avoid low-volume Asian session noise.
# Target: 15-35 trades/year per symbol to minimize fee drag.
# Works in bull markets via upside breakouts and bear markets via downside breakdowns.

name = "1h_Donchian20_4hTrend_VolumeSpike_Session"
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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels (trend filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period) from prior completed 4h bar
    donchian_h_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_l_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_h_4h_shifted = np.roll(donchian_h_4h, 1)
    donchian_l_4h_shifted = np.roll(donchian_l_4h, 1)
    donchian_h_4h_shifted[0] = np.nan
    donchian_l_4h_shifted[0] = np.nan
    
    # Align to 1h timeframe (already delayed by align_htf_to_ltf for completed bar)
    donchian_h_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_h_4h_shifted)
    donchian_l_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_l_4h_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if not in trading session or any value is NaN
        if not in_session[i] or \
           (np.isnan(donchian_h_4h_aligned[i]) or np.isnan(donchian_l_4h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high AND volume spike
            if close[i] > donchian_h_4h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low AND volume spike
            elif close[i] < donchian_l_4h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below 4h Donchian low
            if close[i] < donchian_l_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above 4h Donchian high
            if close[i] > donchian_h_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals