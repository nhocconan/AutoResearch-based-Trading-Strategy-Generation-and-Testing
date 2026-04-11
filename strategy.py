#!/usr/bin/env python3
# 4h_12h_camarilla_breakout_volume_v1
# Strategy: 4-hour Camarilla pivot breakout with volume confirmation and 12-hour trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (H4/L4) act as strong support/resistance. 
# Breakouts above H4 or below L4 with volume confirmation and 12h trend alignment 
# provide high-probability entries. Works in bull markets via upside breakouts and 
# in bear markets via downside breakdowns. Volume filter reduces false breakouts.
# 12h trend filter ensures trades align with higher-timeframe momentum.
# Target: 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volume_v1"
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
    
    # Load 12h data ONCE before loop for Camarilla pivot levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h Camarilla pivot levels (based on previous day's OHLC)
    # Using 12h high/low/close as proxy for daily pivot calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels: H4/L4 = close ± 1.1*(high-low)/2
    # These are the key breakout levels
    hl_range = high_12h - low_12h
    camarilla_h4 = close_12h + 1.1 * hl_range / 2
    camarilla_l4 = close_12h - 1.1 * hl_range / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    # 12h EMA(25) for trend filter
    ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    # 4h Volume confirmation: current volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_25_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_h4_aligned[i]
        breakout_down = close[i] < camarilla_l4_aligned[i]
        
        # Trend filter: price above/below 12h EMA25
        uptrend = close[i] > ema_25_12h_aligned[i]
        downtrend = close[i] < ema_25_12h_aligned[i]
        
        # Entry logic: Breakout + volume + trend alignment
        if breakout_up and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to midpoint of Camarilla range (neutral zone)
        elif position == 1 and close[i] < (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals