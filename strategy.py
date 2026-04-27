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
    
    # Get 4h data for trend and volume context
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMA(34) for trend
    ema_34_4h = np.full(len(df_4h), np.nan)
    alpha_4h = 2 / (34 + 1)
    for i in range(len(close_4h)):
        if i < 33:
            ema_34_4h[i] = np.mean(close_4h[:i+1]) if i > 0 else close_4h[i]
        else:
            if np.isnan(ema_34_4h[i-1]):
                ema_34_4h[i] = np.mean(close_4h[i-33:i+1])
            else:
                ema_34_4h[i] = close_4h[i] * alpha_4h + ema_34_4h[i-1] * (1 - alpha_4h)
    
    # Calculate 4h volume SMA(20) for volume filter
    vol_sma_20_4h = np.full(len(df_4h), np.nan)
    for i in range(len(volume_4h)):
        if i < 19:
            vol_sma_20_4h[i] = np.mean(volume_4h[:i+1]) if i > 0 else volume_4h[i]
        else:
            vol_sma_20_4h[i] = np.mean(volume_4h[i-19:i+1])
    
    # Align 4h indicators to 1h
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    vol_sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_sma_20_4h)
    
    # Calculate 1h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_14 = np.full(n, np.nan)
    rsi_14[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    # Calculate 1h volume ratio for entry confirmation
    vol_sma_20_1h = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            vol_sma_20_1h[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            vol_sma_20_1h[i] = np.mean(volume[i-19:i+1])
    
    volume_ratio = np.full(n, np.nan)
    valid_vol = (~np.isnan(vol_sma_20_1h)) & (vol_sma_20_1h > 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_sma_20_1h[valid_vol]
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(vol_sma_20_4h_aligned[i]) or
            np.isnan(rsi_14[i]) or
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 4h average volume
        vol_filter = volume_ratio[i] > 1.5
        
        if position == 0:
            # Long: price above 4h EMA34 (uptrend) + RSI < 40 (pullback) + volume spike
            if (close[i] > ema_34_4h_aligned[i] and 
                rsi_14[i] < 40 and 
                vol_filter):
                signals[i] = 0.20
                position = 1
            # Short: price below 4h EMA34 (downtrend) + RSI > 60 (bounce) + volume spike
            elif (close[i] < ema_34_4h_aligned[i] and 
                  rsi_14[i] > 60 and 
                  vol_filter):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 60 or price crosses below 4h EMA34
            if (rsi_14[i] > 60 or 
                close[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI < 40 or price crosses above 4h EMA34
            if (rsi_14[i] < 40 or 
                close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA34_RSI_VolumeFilter_Session_v1"
timeframe = "1h"
leverage = 1.0