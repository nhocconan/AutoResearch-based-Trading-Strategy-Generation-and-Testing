# 6h_Aroon32_1dATR_Breakout_Volume1.5x_ATRTrail_2.0x
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d ATR filter (30-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_1d_avg = np.mean(atr_1d[~np.isnan(atr_1d)])
    
    # === 6h Aroon (32-period) ===
    # Aroon Up: measures how recent the highest high was
    # Aroon Down: measures how recent the lowest low was
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(32, n):
        window_high = high[i-32:i+1]
        window_low = low[i-32:i+1]
        high_idx = np.argmax(window_high)  # 0 to 32
        low_idx = np.argmin(window_low)    # 0 to 32
        aroon_up[i] = ((32 - high_idx) / 32) * 100
        aroon_down[i] = ((32 - low_idx) / 32) * 100
    
    # === 1d Volume filter (24-period average of 1h data) ===
    # Since we're on 6h, 4 periods = 1 day
    vol_ma_6h = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(aroon_up[i]) or 
            np.isnan(aroon_down[i]) or
            np.isnan(vol_ma_6h[i]) or
            np.isnan(atr_1d[i]) or
            atr_1d_avg == 0):
            signals[i] = 0.0
            position = 0
            continue
        
        # Conditions
        strong_uptrend = aroon_up[i] > 70 and aroon_down[i] < 30
        strong_downtrend = aroon_down[i] > 70 and aroon_up[i] < 30
        vol_filter = volume[i] > vol_ma_6h[i] * 1.5
        atr_filter = atr_1d[i] > atr_1d_avg * 0.5  # Ensure sufficient volatility
        
        # Exit conditions
        if position == 1 and (not strong_uptrend or not vol_filter):
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and (not strong_downtrend or not vol_filter):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic (only when flat)
        if position == 0:
            if strong_uptrend and vol_filter and atr_filter:
                signals[i] = 0.25
                position = 1
                continue
            elif strong_downtrend and vol_filter and atr_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Aroon32_1dATR_Breakout_Volume1.5x_ATRTrail_2.0x"
timeframe = "6h"
leverage = 1.0