#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray power with 12h trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13, works in both bull/bear markets.
# Combined with 12h EMA trend filter and volume confirmation to reduce false signals.
# Target: 50-150 total trades over 4 years (12-37/year), size 0.25.
name = "6h_ElderRay_12hEMA_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_period = 13
    ema13 = pd.Series(close).ewm(span=ema13_period, adjust=False, min_periods=ema13_period).values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    # 12h EMA(34) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema13[i]) or np.isnan(vol_ema20[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: bull power > 0 + price > 12h EMA34 + volume confirmation
            if (bull_power[i] > 0 and price > ema_34_12h_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: bear power < 0 + price < 12h EMA34 + volume confirmation
            elif (bear_power[i] < 0 and price < ema_34_12h_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bull power <= 0 (loss of bullish momentum)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bear power >= 0 (loss of bearish momentum)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals