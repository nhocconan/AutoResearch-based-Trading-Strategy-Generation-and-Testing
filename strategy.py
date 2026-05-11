#!/usr/bin/env python3
# 1d_1w_Camarilla_R1_S1_Breakout_Trend_Volume
# Hypothesis: Uses daily chart with weekly trend filter. Daily Camarilla R1/S1 breakouts with volume confirmation.
# In bull markets, weekly uptrend + daily breakout above R1 captures momentum.
# In bear markets, weekly downtrend + daily breakdown below S1 captures moves.
# Volume filter ensures breakouts have conviction, reducing false signals.
# Target: 10-25 trades/year to minimize fee drag while capturing meaningful moves.

name = "1d_1w_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly EMA50 for trend filter ---
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # --- Daily Camarilla levels (R1, S1) from previous day ---
    prev_1d_high = high
    prev_1d_low = low
    prev_1d_close = close
    
    # Calculate daily ranges for Camarilla
    daily_range = high - low
    camarilla_width = daily_range * 1.1 / 2.0
    camarilla_r1 = close + camarilla_width
    camarilla_s1 = close - camarilla_width
    
    # Shift to get previous day's levels (available at open)
    camarilla_r1_prev = np.roll(camarilla_r1, 1)
    camarilla_s1_prev = np.roll(camarilla_s1, 1)
    camarilla_r1_prev[0] = np.nan
    camarilla_s1_prev[0] = np.nan
    
    # --- Volume confirmation (2x 20-period average on daily) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for weekly EMA50 (50 weeks ~ 350 days) and 20-period volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1_prev[i]) or
            np.isnan(camarilla_s1_prev[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume surge and weekly uptrend
            if (close[i] > camarilla_r1_prev[i] and 
                volume_surge and 
                ema_50_1w_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume surge and weekly downtrend
            elif (close[i] < camarilla_s1_prev[i] and 
                  volume_surge and 
                  ema_50_1w_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S1 OR weekly EMA50 turns down
                if (close[i] < camarilla_s1_prev[i] or 
                    close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R1 OR weekly EMA50 turns up
                if (close[i] > camarilla_r1_prev[i] or 
                    close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals