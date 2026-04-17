#!/usr/bin/env python3
"""
1h EMA20 + RSI14 + Volume Spike + 4h Trend Filter
Long when EMA20 > RSI14 (50) + volume > 2x 20-bar average + price > 4h EMA50.
Short when EMA20 < RSI14 (50) + volume > 2x average + price < 4h EMA50.
Exit on opposite EMA/RSI cross.
Designed for 1h to capture momentum with volume confirmation and 4h trend filter.
Target: 60-150 trades over 4 years.
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
    
    # EMA20 on 1h
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # RSI14 on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume MA20 on 1h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA20, RSI14
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(ema_50_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        ema_fast = ema_20[i]
        rsi_val = rsi[i]
        trend = ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: EMA20 > RSI50 + volume spike + price > 4h EMA50
            if ema_fast > 50 and rsi_val > 50 and \
               vol > 2.0 * vol_ma and price > trend:
                signals[i] = 0.20
                position = 1
            # Short: EMA20 < RSI50 + volume spike + price < 4h EMA50
            elif ema_fast < 50 and rsi_val < 50 and \
                 vol > 2.0 * vol_ma and price < trend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: EMA20 < RSI50
            if ema_fast < 50 and rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: EMA20 > RSI50
            if ema_fast > 50 and rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA20_RSI14_VolumeSpike_4hTrend"
timeframe = "1h"
leverage = 1.0