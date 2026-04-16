#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) with 1d EMA50 trend filter and volume confirmation
# Long when Williams %R crosses above -20 from below AND price > 1d EMA50 AND volume > 1.5x 24-period avg volume
# Short when Williams %R crosses below -80 from above AND price < 1d EMA50 AND volume > 1.5x 24-period avg volume
# Williams %R captures momentum extremes; EMA50 filters for trend alignment; volume confirms conviction
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === 6h Williams %R (14-period) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Williams %R = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # === 6h Volume Confirmation (24-period average = 6 hours) ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        wr = williams_r[i]
        vol_confirm = volume[i] > vol_ma_24[i] * 1.5  # 1.5x average volume for confirmation
        
        # Williams %R crossover signals
        wr_prev = williams_r[i-1]
        
        # Long when: Williams %R crosses above -20 from below AND price > EMA50 AND volume confirmation
        if wr > -20 and wr_prev <= -20 and price > ema_val and vol_confirm:
            signals[i] = 0.25
            position = 1
            continue
        # Short when: Williams %R crosses below -80 from above AND price < EMA50 AND volume confirmation
        elif wr < -80 and wr_prev >= -80 and price < ema_val and vol_confirm:
            signals[i] = -0.25
            position = -1
            continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR14_1dEMA50_Volume1.5x"
timeframe = "6h"
leverage = 1.0