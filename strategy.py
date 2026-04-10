#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and 1w trend filter (Elder Ray)
# - Long when price breaks above 12h Donchian upper band (20-period high) AND 1d volume > 1.5x 20-period volume SMA AND 1w Elder Ray bull power > 0
# - Short when price breaks below 12h Donchian lower band (20-period low) AND 1d volume > 1.5x 20-period volume SMA AND 1w Elder Ray bear power < 0
# - Exit: price retreats to 12h Donchian midpoint or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 12h timeframe to stay within fee drag limits
# - Uses Donchian structure from 12h, volume confirmation from 1d, trend filter from 1w Elder Ray

name = "12h_1d_1w_donchian_volume_elder_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate 12h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2.0
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1w Elder Ray for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1w = high_1w - ema_13_1w
    bear_power_1w = low_1w - ema_13_1w
    
    # Align 1w Elder Ray to 12h timeframe
    bull_power_1w_aligned = align_htf_to_ltf(prices, df_1w, bull_power_1w)
    bear_power_1w_aligned = align_htf_to_ltf(prices, df_1w, bear_power_1w)
    
    # Align 1d volume SMA to 12h timeframe
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(bull_power_1w_aligned[i]) or np.isnan(bear_power_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA AND 1d volume > 1.5x 20-period volume SMA
        vol_confirm_12h = volume[i] > 1.5 * pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        vol_confirm_1d = volume_1d[i // 4] > 1.5 * volume_sma_20_1d_aligned[i] if i // 4 < len(volume_1d) else False
        vol_confirm = vol_confirm_12h and vol_confirm_1d
        
        # Trend filter: 1w Elder Ray
        trend_bullish = bull_power_1w_aligned[i] > 0
        trend_bearish = bear_power_1w_aligned[i] < 0
        
        # Donchian breakout signals
        breakout_up = close[i] > high_max_20[i-1]  # Break above previous upper band
        breakout_down = close[i] < low_min_20[i-1]  # Break below previous lower band
        
        # Exit conditions: price retreats to midpoint or loss of volume confirmation
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