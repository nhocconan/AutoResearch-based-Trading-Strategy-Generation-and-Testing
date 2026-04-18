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
    
    # Get 4h data for trend and momentum
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h RSI(14) for momentum
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    # Wilder smoothing for RSI
    for i in range(len(gain)):
        if i < 14:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (13) + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * (13) + loss[i]) / 14
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi_4h = np.full_like(close_4h, np.nan)
    valid = avg_loss != 0
    rsi_4h[valid] = 100 - (100 / (1 + avg_gain[valid] / avg_loss[valid]))
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 2 + ema_50[i-1] * 49) / 51
    
    # Align indicators to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike detection: current volume > 2.0x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    volume_spike = volume > 2.0 * vol_ma
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Entry conditions
        rsi_oversold = rsi_4h_aligned[i] < 30
        rsi_overbought = rsi_4h_aligned[i] > 70
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        # Long: RSI oversold + price above EMA + volume spike
        if rsi_oversold and price_above_ema and vol_spike:
            signals[i] = 0.20
        # Short: RSI overbought + price below EMA + volume spike
        elif rsi_overbought and price_below_ema and vol_spike:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_RSI_EMA_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0