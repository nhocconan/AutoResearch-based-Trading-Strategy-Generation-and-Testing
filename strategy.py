#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA34 trend filter and volume spike confirmation.
# Donchian(20) identifies volatility breakouts; EMA34 on 1d confirms trend direction.
# Volume > 1.5x average confirms institutional participation. Designed for low trade frequency (<40/year).
name = "12h_Donchian20_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) channels on 12h timeframe
    # Use rolling window of 20 periods for high and low
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    
    for i in range(20, n):
        high_max[i] = np.max(high[i-20:i])
        low_min[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need 34 for EMA34 and 20 for Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_34_1d_aligned[i]
        upper_channel = high_max[i]
        lower_channel = low_min[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Close > Upper Donchian AND price > 1d EMA34 (uptrend) AND volume > 1.5x average
            if close[i] > upper_channel and close[i] > ema_1d and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < Lower Donchian AND price < 1d EMA34 (downtrend) AND volume > 1.5x average
            elif close[i] < lower_channel and close[i] < ema_1d and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < Lower Donchian OR trend reverses (price < 1d EMA34)
            if close[i] < lower_channel or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > Upper Donchian OR trend reverses (price > 1d EMA34)
            if close[i] > upper_channel or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals