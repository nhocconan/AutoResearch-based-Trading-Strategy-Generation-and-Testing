#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w trend filter
# - Long when price breaks above 4h Donchian upper band (20-period high) AND 1d volume > 1.2x 20-period volume SMA AND 1w close > 1w EMA20
# - Short when price breaks below 4h Donchian lower band (20-period low) AND 1d volume > 1.2x 20-period volume SMA AND 1w close < 1w EMA20
# - Exit: price retraces to midpoint of Donchian channel or volume confirmation lost
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
# - Uses Donchian structure from 4h for breakouts, 1d for volume confirmation, 1w for trend filter

name = "4h_1d_1w_donchian_volume_trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2.0
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1w close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.2x 20-period volume SMA
        # Get current 1d volume index (4h bars per 1d = 6)
        idx_1d = i // 6
        if idx_1d >= len(volume_1d):
            vol_confirm = False
        else:
            vol_confirm = volume_1d[idx_1d] > 1.2 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1w close vs 1w EMA20
        trend_bullish = close_1w_aligned[i] > ema_20_1w_aligned[i]
        trend_bearish = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > high_max_20[i-1]  # Break above previous upper band
        breakout_down = close[i] < low_min_20[i-1]  # Break below previous lower band
        
        # Exit conditions: price retraces to midpoint or loss of volume confirmation
        exit_long = close[i] < donchian_mid[i] or not vol_confirm
        exit_short = close[i] > donchian_mid[i] or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and trend_bearish and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals