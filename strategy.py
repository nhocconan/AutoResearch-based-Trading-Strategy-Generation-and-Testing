#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w EMA34 trend filter
# Long when price breaks above upper Donchian(20) AND volume > 1.8x 20-period average AND 1w EMA34 > EMA34_prev (uptrend)
# Short when price breaks below lower Donchian(20) AND volume > 1.8x 20-period average AND 1w EMA34 < EMA34_prev (downtrend)
# Exit when price crosses back to the midpoint of the Donchian channel OR 1w EMA34 flips direction
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-25 trades/year per symbol.
# Donchian channels provide clear breakout levels, volume spike confirms institutional participation,
# 1w EMA34 filters for primary trend to avoid counter-trend whipsaws in bear markets.

name = "1d_Donchian20_VolumeSpike_1wEMA34_Trend"
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
    
    # Get 1w data ONCE before loop for EMA34 and volume calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w data
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_prev = np.concatenate([[np.nan], ema_34[:-1]])  # Previous EMA for trend direction
    
    # Uptrend when current EMA34 > previous EMA34
    uptrend_1w = ema_34 > ema_34_prev
    downtrend_1w = ema_34 < ema_34_prev
    
    # Align 1w trend to 1d timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume confirmation: volume > 1.8x 20-period average (spike filter) on 1w data
    volume_1w = df_1w['volume'].values
    if len(volume_1w) >= 20:
        vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
        volume_filter_1w = volume_1w > (1.8 * vol_ma_20)
    else:
        volume_filter_1w = np.zeros(len(volume_1w), dtype=bool)
    
    volume_filter_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_filter_1w.astype(float))
    
    # Calculate Donchian(20) channels on 1d data (using historical data only)
    if len(high) >= 20:
        # Upper channel: highest high over past 20 periods (excluding current)
        upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
        # Lower channel: lowest low over past 20 periods (excluding current)
        lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
        # Midpoint for exit
        midpoint = (upper_channel + lower_channel) / 2
    else:
        upper_channel = np.full(n, np.nan)
        lower_channel = np.full(n, np.nan)
        midpoint = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_filter_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian AND volume spike AND 1w uptrend
            if (close[i] > upper_channel[i] and 
                volume_filter_1w_aligned[i] > 0.5 and 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian AND volume spike AND 1w downtrend
            elif (close[i] < lower_channel[i] and 
                  volume_filter_1w_aligned[i] > 0.5 and 
                  downtrend_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to midpoint OR 1w trend flips to downtrend
            if (close[i] < midpoint[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to midpoint OR 1w trend flips to uptrend
            if (close[i] > midpoint[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals