#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA21 trend filter + volume spike
# - Long when price breaks above Donchian upper(20) on 4h AND price > 12h EMA21 AND volume > 1.5x 20-period average
# - Short when price breaks below Donchian lower(20) on 4h AND price < 12h EMA21 AND volume > 1.5x 20-period average
# - Exit when price crosses back through Donchian midpoint (mean of upper/lower)
# - Donchian captures breakouts, EMA21 filters for trend alignment, volume spike confirms conviction
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 19-50 trades per year per symbol (75-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for EMA21 calculation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA21 on 12h timeframe
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 12h EMA21 to 4h timeframe
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate Donchian channels on 4h timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    # Calculate volume spike (current volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or np.isnan(ema_21_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = highest_high_20[i]
        lower = lowest_low_20[i]
        mid = donchian_mid[i]
        ema21 = ema_21_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long entry: price breaks above upper AND price > EMA21 AND volume spike
            if price > upper and price > ema21 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower AND price < EMA21 AND volume spike
            elif price < lower and price < ema21 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below midpoint
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midpoint
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA21_VolumeSpike"
timeframe = "4h"
leverage = 1.0