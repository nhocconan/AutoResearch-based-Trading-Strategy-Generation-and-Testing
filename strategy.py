#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w EMA34 trend filter
# Long when price breaks above Donchian upper band AND volume > 2.0x 20-period average AND 1w EMA34 > EMA34_prev (uptrend)
# Short when price breaks below Donchian lower band AND volume > 2.0x 20-period average AND 1w EMA34 < EMA34_prev (downtrend)
# Exit when price crosses back to Donchian mid-band OR 1w EMA34 flips direction
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Donchian provides price channel structure, volume spike confirms breakout strength,
# 1w EMA34 filters for primary trend to avoid counter-trend whipsaws in bear markets.

name = "12h_Donchian20_VolumeSpike_1wEMA34_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 and trend direction
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_prev = np.concatenate([[np.nan], ema_34[:-1]])  # Previous EMA for trend direction
    uptrend_1w = ema_34 > ema_34_prev
    downtrend_1w = ema_34 < ema_34_prev
    
    # Align 1w trend to 12h timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average (using 1d volume)
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike_1d = vol_1d > (2.0 * vol_ma_20)
    else:
        volume_spike_1d = np.zeros(len(df_1d), dtype=bool)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate Donchian(20) on 12h data (using lookback period to avoid look-ahead)
    # We calculate the Donchian bands based on the last 20 completed periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND volume spike AND 1w uptrend
            if (close[i] > donchian_upper[i] and 
                volume_spike_aligned[i] > 0.5 and 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND volume spike AND 1w downtrend
            elif (close[i] < donchian_lower[i] and 
                  volume_spike_aligned[i] > 0.5 and 
                  downtrend_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to Donchian mid-band OR 1w trend flips to downtrend
            if (close[i] < donchian_mid[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to Donchian mid-band OR 1w trend flips to uptrend
            if (close[i] > donchian_mid[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals