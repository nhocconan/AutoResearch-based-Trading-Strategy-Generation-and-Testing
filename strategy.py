#!/usr/bin/env python3
"""
1h EMA13 Trend + 4h RSI(14) Filter + Volume Spike
Long: EMA13 rising + 4h RSI > 50 + volume > 1.5x 1h volume SMA(20)
Short: EMA13 falling + 4h RSI < 50 + volume > 1.5x 1h volume SMA(20)
Exit: Opposite EMA13 direction or RSI crosses 50
Uses EMA for fast trend filtering with 4h RSI to avoid counter-trend trades.
Volume spike confirms momentum. Designed for 1h timeframe with controlled trade frequency.
Target: 60-150 total trades over 4 years (15-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h RSI(14)
    delta = pd.Series(close_4h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_4h = 100 - (100 / (1 + rs))
    rsi_14_4h = rsi_14_4h.fillna(50).values  # neutral when undefined
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # Calculate EMA13 for trend
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1h volume SMA(20)
    vol_sma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(20, 13)  # need volume SMA and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_14_4h_aligned[i]) or np.isnan(vol_sma_1h[i]) or
            np.isnan(ema13[i]) or np.isnan(ema13[i-1])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_1h[i]
        rsi_val = rsi_14_4h_aligned[i]
        ema_val = ema13[i]
        ema_prev = ema13[i-1]
        
        if position == 0:
            # Long: EMA13 rising + 4h RSI > 50 + volume spike
            if ema_val > ema_prev and rsi_val > 50 and vol > 1.5 * vol_sma_val:
                signals[i] = 0.20
                position = 1
            # Short: EMA13 falling + 4h RSI < 50 + volume spike
            elif ema_val < ema_prev and rsi_val < 50 and vol > 1.5 * vol_sma_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: EMA13 falling or 4h RSI < 50
            if ema_val < ema_prev or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: EMA13 rising or 4h RSI > 50
            if ema_val > ema_prev or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA13_Trend_4hRSI_VolumeSpike"
timeframe = "1h"
leverage = 1.0