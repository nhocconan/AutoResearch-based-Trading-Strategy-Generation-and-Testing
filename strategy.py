#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_volume_v1
# Strategy: 1-day Camarilla pivot breakout with volume confirmation and weekly trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels on daily charts identify key support/resistance levels. 
# Breakouts above resistance (H4) or below support (L4) with volume confirmation and aligned weekly trend 
# capture institutional moves. Weekly EMA filter ensures we only trade in the direction of the higher timeframe trend.
# Low trade frequency (target: 10-25 trades/year) to minimize fee drag and improve generalization.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Camarilla pivot levels (based on previous day's range)
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Calculate pivot levels using previous day's data
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First day will have NaN due to roll, that's expected
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    H4 = pivot + (range_val * 1.1 / 2)  # Resistance 4
    L4 = pivot - (range_val * 1.1 / 2)  # Support 4
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid (NaN from roll or calculations)
        if (np.isnan(H4[i]) or np.isnan(L4[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            # Hold current position or flat
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        # Trend filter: price above/below weekly EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + trend alignment
        if close[i] > H4[i] and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] < L4[i] and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to pivot point
        elif position == 1 and close[i] < pivot[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > pivot[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals