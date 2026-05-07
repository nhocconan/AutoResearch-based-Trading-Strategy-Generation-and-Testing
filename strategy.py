#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3)
    # Pivot = (H + L + C) / 3
    # R3 = Close + (High - Low) * 1.1 / 2
    # S3 = Close - (High - Low) * 1.1 / 2
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    r3 = df_1d['close'] + hl_range * 1.1 / 2
    s3 = df_1d['close'] - hl_range * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = df_1d['close'].ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Wait for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R3, price above EMA34, volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume[i] > vol_ma[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # Short: Close < S3, price below EMA34, volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume[i] > vol_ma[i] * 1.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S3 or price below EMA34
            if (close[i] < s3_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R3 or price above EMA34
            if (close[i] > r3_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Camarilla levels provide precise support/resistance based on previous day's range.
# Breakout above R3 with trend alignment (price > EMA34) signals bullish momentum.
# Breakdown below S3 with trend alignment (price < EMA34) signals bearish momentum.
# Volume confirmation ensures institutional participation in the breakout.
# Works in bull markets (buy R3 breaks in uptrend) and bear markets (sell S3 breaks in downtrend).
# Position size 0.25 balances risk and keeps trade frequency ~20-50/year.