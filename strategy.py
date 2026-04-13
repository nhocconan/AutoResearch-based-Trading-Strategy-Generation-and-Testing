#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume spike
# RSI < 30 for long, RSI > 70 for short in ranging markets
# 4h EMA50 as trend filter: only long when price > EMA50, short when price < EMA50
# Volume spike (2x average) confirms momentum
# Session filter: 08-20 UTC to avoid low-volume Asian session
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Works in both bull/bear: mean reversion in ranges, trend filter avoids counter-trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # Average volume (20-period)
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = np.zeros(len(close_4h))
    for i in range(50, len(close_4h)):
        if i == 50:
            ema_4h[i] = np.mean(close_4h[0:51])
        else:
            ema_4h[i] = (close_4h[i] * (2 / (50 + 1))) + (ema_4h[i-1] * (1 - (2 / (50 + 1))))
    
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    session_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        session_mask[i] = (8 <= hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        rsi_val = rsi[i]
        ema_4h_val = ema_4h_aligned[i]
        
        # Volume spike: current volume > 2x average volume
        volume_spike = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price above 4h EMA50 + volume spike
            if (rsi_val < 30 and 
                price > ema_4h_val and
                volume_spike):
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) + price below 4h EMA50 + volume spike
            elif (rsi_val > 70 and 
                  price < ema_4h_val and
                  volume_spike):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) OR volume drops
            if (rsi_val > 50 or 
                vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) OR volume drops
            if (rsi_val < 50 or 
                vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_RSI_MeanReversion_Volume_Spike_v1"
timeframe = "1h"
leverage = 1.0