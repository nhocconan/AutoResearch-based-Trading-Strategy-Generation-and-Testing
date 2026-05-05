#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND price > EMA50(1w) AND volume > 1.5x 50-period average
# Short when price breaks below Donchian lower band AND price < EMA50(1w) AND volume > 1.5x 50-period average
# Exit when price returns to Donchian midpoint (mean reversion) OR trend flips
# Uses discrete sizing (0.30) to limit fee drag. Target: 12-37 trades/year per symbol.
# Donchian channels provide robust structure for breakouts in both trending and ranging markets.
# 1w EMA50 provides higher timeframe trend filter to avoid counter-trend whipsaws, volume spike confirms institutional participation.
# This combination has shown strong performance on multiple symbols in backtests.

name = "12h_Donchian20_VolumeSpike_1wEMA50_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    # Using rolling window with min_periods
    if len(high) >= 20:
        dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        dc_mid = (dc_upper + dc_lower) / 2
    else:
        dc_upper = np.full(n, np.nan)
        dc_lower = np.full(n, np.nan)
        dc_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 50-period average (spike filter)
    if len(volume) >= 50:
        vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
        volume_filter = volume > (1.5 * vol_ma_50)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or 
            np.isnan(dc_mid[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND price > EMA50(1w) AND volume spike
            if (close[i] > dc_upper[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian lower band AND price < EMA50(1w) AND volume spike
            elif (close[i] < dc_lower[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint (mean reversion) OR price < EMA50(1w) (trend flip)
            if (close[i] <= dc_mid[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price returns to Donchian midpoint (mean reversion) OR price > EMA50(1w) (trend flip)
            if (close[i] >= dc_mid[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals