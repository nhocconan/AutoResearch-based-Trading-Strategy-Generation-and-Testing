#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume spike confirmation
# Long when price breaks above 20-period Donchian high AND 12h EMA50 > EMA50_prev (uptrend) AND volume > 2.0x 20-period average
# Short when price breaks below 20-period Donchian low AND 12h EMA50 < EMA50_prev (downtrend) AND volume > 2.0x 20-period average
# Exit when price crosses back to the Donchian midpoint OR 12h EMA50 flips direction
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Donchian channels provide clear breakout levels, volume spike confirms institutional participation,
# 12h EMA50 filters for primary trend direction to avoid counter-trend whipsaws.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "6h_Donchian20_VolumeSpike_12hEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian levels and EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 12h data (using previous period's data to avoid look-ahead)
    prev_high = np.concatenate([[np.nan], df_12h['high'].values[:-1]])
    prev_low = np.concatenate([[np.nan], df_12h['low'].values[:-1]])
    
    # Donchian high: highest high over last 20 periods
    donchian_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over last 20 periods
    donchian_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Get 12h data for EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_prev = np.concatenate([[np.nan], ema_50[:-1]])  # Previous EMA for trend direction
    
    # Uptrend when current EMA50 > previous EMA50
    uptrend_12h = ema_50 > ema_50_prev
    downtrend_12h = ema_50 < ema_50_prev
    
    # Align 12h trend to 6h timeframe
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h.astype(float))
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(uptrend_12h_aligned[i]) or 
            np.isnan(downtrend_12h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume spike AND 12h uptrend
            if (close[i] > donchian_high_aligned[i] and 
                volume_filter[i] and 
                uptrend_12h_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND volume spike AND 12h downtrend
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_filter[i] and 
                  downtrend_12h_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to Donchian midpoint OR 12h trend flips to downtrend
            if (close[i] < donchian_mid_aligned[i] or 
                downtrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to Donchian midpoint OR 12h trend flips to uptrend
            if (close[i] > donchian_mid_aligned[i] or 
                uptrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals