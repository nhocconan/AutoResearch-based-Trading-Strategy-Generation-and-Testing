#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w EMA50 trend filter
# Long when price breaks above 20-period high AND volume > 2.0x 20-period average AND 1w EMA50 > EMA50_prev (uptrend)
# Short when price breaks below 20-period low AND volume > 2.0x 20-period average AND 1w EMA50 < EMA50_prev (downtrend)
# Exit when price crosses back to the opposite Donchian level OR 1w EMA50 flips direction
# Uses discrete sizing (0.25) to limit fee drag. Target: 7-25 trades/year per symbol.
# Donchian provides clear price channels, volume spike confirms institutional interest,
# 1w EMA50 filters for primary trend direction to avoid counter-trend whipsaws.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "1d_Donchian20_VolumeSpike_1wEMA50_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for trend filter and volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_prev = np.concatenate([[np.nan], ema_50[:-1]])  # Previous EMA for trend direction
    
    # Uptrend when current EMA50 > previous EMA50
    uptrend_1w = ema_50 > ema_50_prev
    downtrend_1w = ema_50 < ema_50_prev
    
    # Align 1w trend to 1d timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter) on 1w data
    volume_1w = df_1w['volume'].values
    if len(volume_1w) >= 20:
        vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
        volume_filter_1w = volume_1w > (2.0 * vol_ma_20_1w)
    else:
        volume_filter_1w = np.zeros(len(df_1w), dtype=bool)
    
    # Align 1w volume filter to 1d timeframe
    volume_filter_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_filter_1w.astype(float))
    
    # Calculate Donchian channels on 1d data (20-period high/low)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_filter_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume spike AND 1w uptrend
            if (close[i] > donchian_high[i] and 
                volume_filter_1w_aligned[i] > 0.5 and 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND volume spike AND 1w downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_filter_1w_aligned[i] > 0.5 and 
                  downtrend_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to Donchian low OR 1w trend flips to downtrend
            if (close[i] < donchian_low[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to Donchian high OR 1w trend flips to uptrend
            if (close[i] > donchian_high[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals