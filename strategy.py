#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with volume spike and 1d trend filter
# Long when price touches Camarilla L3 support + volume spike + 1d uptrend
# Short when price touches Camarilla H3 resistance + volume spike + 1d downtrend
# Exit when price reaches Camarilla L4/H4 or trend reverses
# Designed for 12-37 trades/year on 12h timeframe with mean reversion in ranging markets and trend alignment

name = "12h_1d_camarilla_reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous 12h bar (H4, L3, H3, L4)
    # Camarilla: Close +- (High-Low) * 1.1/12, 1.1/6, 1.1/4, 1.1/2
    cam_H4 = np.zeros(n)
    cam_L3 = np.zeros(n)
    cam_H3 = np.zeros(n)
    cam_L4 = np.zeros(n)
    
    for i in range(1, n):
        if not (np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(close[i-1])):
            range_ = high[i-1] - low[i-1]
            cam_H4[i] = close[i-1] + range_ * 1.1 / 2
            cam_L3[i] = close[i-1] - range_ * 1.1 / 6
            cam_H3[i] = close[i-1] + range_ * 1.1 / 4
            cam_L4[i] = close[i-1] - range_ * 1.1 / 12
        else:
            cam_H4[i] = cam_H4[i-1] if i > 1 else 0
            cam_L3[i] = cam_L3[i-1] if i > 1 else 0
            cam_H3[i] = cam_H3[i-1] if i > 1 else 0
            cam_L4[i] = cam_L4[i-1] if i > 1 else 0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(cam_L3[i]) or np.isnan(cam_H3[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA50
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: price touches Camarilla levels with volume and trend
        long_entry = (close[i] <= cam_L3[i] * 1.001) and volume_filter and is_uptrend  # Allow small buffer
        short_entry = (close[i] >= cam_H3[i] * 0.999) and volume_filter and is_downtrend  # Allow small buffer
        
        # Exit conditions: price reaches opposite Camarilla level or trend reverses
        long_exit = (close[i] >= cam_H4[i]) or (not is_uptrend)
        short_exit = (close[i] <= cam_L4[i]) or (not is_downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals