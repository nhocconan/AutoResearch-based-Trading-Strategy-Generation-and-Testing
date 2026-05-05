#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1d EMA50 trend filter
# Long when price breaks above upper Donchian channel AND volume > 1.5x 20-period average AND 1d EMA50 > EMA50_prev (uptrend)
# Short when price breaks below lower Donchian channel AND volume > 1.5x 20-period average AND 1d EMA50 < EMA50_prev (downtrend)
# Exit when price crosses back to the middle of the Donchian channel OR 1d EMA50 flips direction
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-50 trades/year per symbol.
# Donchian channel provides clear breakout levels, volume spike confirms conviction,
# 1d EMA50 filters for primary trend to avoid counter-trend whipsaws in bear markets.

name = "4h_Donchian20_VolumeSpike_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume spike and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channel on 4h data (using 20-period lookback)
    if len(high) >= 20:
        upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
        middle_channel = (upper_channel + lower_channel) / 2
    else:
        upper_channel = np.full(n, np.nan)
        lower_channel = np.full(n, np.nan)
        middle_channel = np.full(n, np.nan)
    
    # Get 1d data for volume spike filter (using previous day's volume to avoid look-ahead)
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike = vol_1d > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d volume spike to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Get 1d data for EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_prev = np.concatenate([[np.nan], ema_50[:-1]])  # Previous EMA for trend direction
    
    # Uptrend when current EMA50 > previous EMA50
    uptrend_1d = ema_50 > ema_50_prev
    downtrend_1d = ema_50 < ema_50_prev
    
    # Align 1d trend to 4h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(middle_channel[i]) or 
            np.isnan(volume_spike_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian AND volume spike AND 1d uptrend
            if (close[i] > upper_channel[i] and 
                volume_spike_aligned[i] > 0.5 and 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian AND volume spike AND 1d downtrend
            elif (close[i] < lower_channel[i] and 
                  volume_spike_aligned[i] > 0.5 and 
                  downtrend_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to middle channel OR 1d trend flips to downtrend
            if (close[i] < middle_channel[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to middle channel OR 1d trend flips to uptrend
            if (close[i] > middle_channel[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals