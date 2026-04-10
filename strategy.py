#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike confirmation and 1w trend filter
# - Long when price breaks above 4h Donchian upper band (20-period high) AND 1d volume > 1.5x 20-period volume SMA AND 1w close > 1w EMA50
# - Short when price breaks below 4h Donchian lower band (20-period low) AND 1d volume > 1.5x 20-period volume SMA AND 1w close < 1w EMA50
# - Exit: price retreats to 4h Donchian midpoint or volume drops below confirmation threshold
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 19-50 trades/year on 4h timeframe to stay within fee drag limits
# - Uses Donchian channels for structure, volume for confirmation, weekly trend for bias

name = "4h_1d_1w_donchian_volume_trend_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian channels (20-period)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll_max
    donchian_lower = low_roll_min
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    for i in range(60, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        # Get 1d volume for current 4h bar (each 1d bar contains 6x 4h bars)
        idx_1d = i // 6
        if idx_1d < len(volume_1d):
            vol_1d_current = volume_1d[idx_1d]
            vol_confirm = vol_1d_current > 1.5 * volume_sma_20_1d_aligned[i]
        else:
            vol_confirm = False
        
        # Trend filter: 1w close vs 1w EMA50
        trend_bullish = close_1w_aligned[i] > ema_50_1w_aligned[i]
        trend_bearish = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # Break above previous upper band
        breakout_down = close[i] < donchian_lower[i-1]  # Break below previous lower band
        
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