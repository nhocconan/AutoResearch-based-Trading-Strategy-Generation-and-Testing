#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w EMA34 trend filter
# Long when price breaks above Donchian upper band AND volume > 1.8x 20-period average AND 1w EMA34 rising
# Short when price breaks below Donchian lower band AND volume > 1.8x 20-period average AND 1w EMA34 falling
# Exit when price crosses back to Donchian midpoint OR 1w EMA34 flips direction
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-50 trades/year per symbol.
# Donchian provides clear structure, volume spike confirms momentum, 1w EMA34 filters for primary trend.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Donchian20_VolumeSpike_1wEMA34_Trend"
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
    
    # Get 1d data ONCE before loop for volume spike calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume average on 1d data
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.8 * vol_ma_20_1d)
    
    # Align 1d volume spike to 4h timeframe
    volume_spike_4h = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_prev = np.concatenate([[np.nan], ema_34[:-1]])  # Previous EMA for trend direction
    
    # Uptrend when current EMA34 > previous EMA34
    uptrend_1w = ema_34 > ema_34_prev
    downtrend_1w = ema_34 < ema_34_prev
    
    # Align 1w trend to 4h timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Calculate Donchian channels on 4h data (using lookback window)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-calculate Donchian levels for efficiency
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    midpoint = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_band[i] = np.max(high[i-20:i])
        lower_band[i] = np.min(low[i-20:i])
        midpoint[i] = (upper_band[i] + lower_band[i]) / 2.0
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(midpoint[i]) or 
            np.isnan(volume_spike_4h[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band AND volume spike AND 1w uptrend
            if (close[i] > upper_band[i] and 
                volume_spike_4h[i] > 0.5 and 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band AND volume spike AND 1w downtrend
            elif (close[i] < lower_band[i] and 
                  volume_spike_4h[i] > 0.5 and 
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