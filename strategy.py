#!/usr/bin/env python3
# 4h_12h_camarilla_volume_v1
# Strategy: 4h Camarilla pivot levels with volume confirmation and 12h trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance. In bull markets, price bounces from L3/S3 and breaks above H3/H4 with volume. In bear markets, price rejects at H3/H4 and breaks below L3/L4 with volume. Uses 12h EMA50 for trend filter to avoid counter-trend trades. Low frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels for each 4h bar using previous bar's H/L/C
    # Camarilla formulas:
    # H4 = C + (H-L) * 1.5
    # H3 = C + (H-L) * 1.25
    # L3 = C - (H-L) * 1.25
    # L4 = C - (H-L) * 1.5
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # Set first bar values to avoid NaN
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    hl_range = prev_high - prev_low
    camarilla_h4 = prev_close + hl_range * 1.5
    camarilla_h3 = prev_close + hl_range * 1.25
    camarilla_l3 = prev_close - hl_range * 1.25
    camarilla_l4 = prev_close - hl_range * 1.5
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.8 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + trend alignment
        if (close[i] > camarilla_h4[i] and vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < camarilla_l4[i] and vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to mean (L3/H3) or trend change
        elif position == 1 and (close[i] < camarilla_h3[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > camarilla_l3[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals