#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: Uses Camarilla pivot points (R1/S1) on 1h for breakout entries, filtered by 4h trend direction and volume spikes.
# Long when: 4h uptrend (close > EMA50), volume > 1.5x 20-period average, and price breaks above R1 level.
# Short when: 4h downtrend (close < EMA50), volume > 1.5x 20-period average, and price breaks below S1 level.
# Exit when price returns to the Camarilla pivot point (PP) or trend reverses.
# Designed to capture intraday breakouts with trend alignment and volume confirmation, effective in both trending and ranging markets.
# Camarilla levels provide precise support/resistance, reducing false breakouts.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend (EMA50) and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4h trend: EMA50 ---
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # --- Camarilla pivot points on 1h (using previous bar) ---
    # Calculate PP, R1, S1 from previous bar's OHLC
    pp = (np.roll(high, 1) + np.roll(low, 1) + np.roll(close, 1)) / 3.0
    r1 = pp + (np.roll(high, 1) - np.roll(low, 1)) * 1.1 / 12.0
    s1 = pp - (np.roll(high, 1) - np.roll(low, 1)) * 1.1 / 12.0
    # First bar: no previous bar, set to NaN
    pp[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(pp[i]) or
            np.isnan(r1[i]) or
            np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend from 4h EMA50
        is_uptrend = close[i] > ema_50_4h_aligned[i]
        is_downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_uptrend and vol_spike:
                # Long: 4h uptrend + volume spike + price breaks above R1
                if close[i] > r1[i]:
                    signals[i] = 0.20
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: 4h downtrend + volume spike + price breaks below S1
                if close[i] < s1[i]:
                    signals[i] = -0.20
                    position = -1
        else:
            if position == 1:
                # Exit long: price returns to PP or trend reverses
                if close[i] <= pp[i] or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price returns to PP or trend reverses
                if close[i] >= pp[i] or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals