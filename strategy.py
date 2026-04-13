# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h trading with 12h directional filter and 1d volume confirmation.
Uses 12h EMA(20) for trend direction, 1d volume spike (volume > 1.8x 20-period average) 
to confirm momentum, and 6h RSI(14) for entry timing. Long when 12h EMA up + 1d volume spike + 6h RSI crosses above 50.
Short when 12h EMA down + 1d volume spike + 6h RSI crosses below 50.
Exit when RSI crosses back to 50 or opposite EMA direction.
Designed for 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
Works in bull/bear: EMA filter adapts to trend, volume confirms strength, RSI provides mean-reversion within trend.
"""

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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(20) for trend direction
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike (volume > 1.8x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.8)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate 6h RSI(14) for entry timing
    # RSI = 100 - (100 / (1 + RS)), RS = average gain / average loss
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA
        ema_up = ema_12h_aligned[i] > ema_12h_aligned[i-1] if i > 0 else False
        ema_down = ema_12h_aligned[i] < ema_12h_aligned[i-1] if i > 0 else False
        
        # Volume confirmation
        vol_confirm = vol_spike_aligned[i] > 0.5
        
        # RSI conditions
        rsi_above_50 = rsi[i] > 50
        rsi_below_50 = rsi[i] < 50
        rsi_cross_up = (i > 0) and (rsi[i-1] <= 50) and (rsi[i] > 50)
        rsi_cross_down = (i > 0) and (rsi[i-1] >= 50) and (rsi[i] < 50)
        
        # Entry conditions
        long_entry = ema_up and vol_confirm and rsi_cross_up
        short_entry = ema_down and vol_confirm and rsi_cross_down
        
        # Exit conditions: RSI crosses back to 50 or trend changes
        exit_long = (position == 1) and (rsi_cross_down or ema_down)
        exit_short = (position == -1) and (rsi_cross_up or ema_up)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_ema_1d_volume_rsi"
timeframe = "6h"
leverage = 1.0