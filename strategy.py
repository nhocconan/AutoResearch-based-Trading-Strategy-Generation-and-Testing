#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike
# Long when: Price breaks above 20-day high AND 1w close > 1w EMA50 AND 1d volume > 1.8x 20-period average
# Short when: Price breaks below 20-day low AND 1w close < 1w EMA50 AND 1d volume > 1.8x 20-period average
# Exit when price touches opposite 20-day level (low for long, high for short)
# Donchian provides clear structure, 1w EMA50 filters for higher-timeframe trend, volume confirms participation
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25

name = "1d_Donchian20_1wEMA50_VolumeSpike_1.8x"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper = max(high over last 20 periods), Lower = min(low over last 20 periods)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d (no shift needed as already 1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1d volume spike (current volume > 1.8x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.8 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1w EMA50 to 1d
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        vol_cond = bool(vol_spike_aligned[i])
        above_ema = close_1d[i] > ema_50_1w_aligned[i]  # 1d close vs 1w EMA50
        below_ema = close_1d[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Break above Donchian high in uptrend with volume spike
            if close_1d[i] > donchian_high_aligned[i] and above_ema and vol_cond:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low in downtrend with volume spike
            elif close_1d[i] < donchian_low_aligned[i] and below_ema and vol_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: touch Donchian low (opposite level)
            if close_1d[i] <= donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: touch Donchian high (opposite level)
            if close_1d[i] >= donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals