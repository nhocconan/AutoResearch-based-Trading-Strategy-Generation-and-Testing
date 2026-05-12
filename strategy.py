#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_12HTREND_VOLUME_SPIKE
# Hypothesis: Camarilla pivot breakout at R1/S1 levels with 12h trend filter (EMA50) and volume spike confirmation. 
# Uses 1d Camarilla levels calculated from previous day's OHLC. Long when price breaks above R1 with volume spike and 12h EMA50 uptrend; short when breaks below S1 with volume spike and downtrend.
# Exit when price returns to cam pivot level (central pivot) or trend reverses.
# Designed for 4h timeframe to capture institutional breakouts with volume confirmation in both bull and bear markets.
# Targets 20-40 trades/year to minimize fee drag with high-probability setups.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_12HTREND_VOLUME_SPIKE"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    # We need to get daily OHLC from the price data
    # Since we're on 4h timeframe, we can resample conceptually but we'll use the close of each day
    # Instead, we calculate Camarilla levels using rolling window of 6 periods (6*4h = 24h)
    # This gives us the previous day's OHLC approximation for each 4h bar
    
    # For each point, we need the OHLC of the previous day
    # We'll use the last completed day's data
    # Simple approach: use rolling window of 6 bars back (24 hours ago) for OHLC
    
    # Shift by 6 periods to get previous day's data (6 * 4h = 24h)
    prev_day_high = np.roll(high, 6)
    prev_day_low = np.roll(low, 6)
    prev_day_close = np.roll(close, 6)
    
    # For first 6 bars, we don't have previous day data, so fill with current
    prev_day_high[:6] = high[:6]
    prev_day_low[:6] = low[:6]
    prev_day_close[:6] = close[:6]
    
    # Calculate Camarilla levels
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    # Pivot = (H+L+C)/3
    hl_range = prev_day_high - prev_day_low
    camarilla_pivot = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    camarilla_r1 = camarilla_pivot + hl_range * 1.1 / 12.0
    camarilla_s1 = camarilla_pivot - hl_range * 1.1 / 12.0
    
    # Volume spike: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 1.5
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    pclose_12h = df_12h['close'].values
    ema50_12h = pd.Series(pclose_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure volume MA and EMA are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(vol_ma[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(camarilla_pivot[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R1 with volume spike and 12h uptrend
            if close[i] > camarilla_r1[i] and vol_spike[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 with volume spike and 12h downtrend
            elif close[i] < camarilla_s1[i] and vol_spike[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to pivot level or trend breaks
            if close[i] < camarilla_pivot[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to pivot level or trend breaks
            if close[i] > camarilla_pivot[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals