#!/usr/bin/env python3
# 4h_1d_Camarilla_R1_S1_Breakout_VolumeSurge_Trend
# Hypothesis: Combines daily trend filter with 4-hour Camarilla R1/S1 breakouts and volume surge confirmation.
# In bull markets, daily uptrend + 4h breakout above R1 captures momentum with tight stops.
# In bear markets, daily downtrend + 4h breakdown below S1 captures accelerated moves.
# Volume surge ensures breakouts have conviction, reducing false signals.
# Target: 20-40 trades/year to minimize fee drag while capturing meaningful moves.

name = "4h_1d_Camarilla_R1_S1_Breakout_VolumeSurge_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1-day data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA34 for trend filter ---
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Daily Camarilla levels (R1, S1) from previous day ---
    prev_1d_high = df_1d['high'].values
    prev_1d_low = df_1d['low'].values
    prev_1d_close = df_1d['close'].values
    
    camarilla_width = (prev_1d_high - prev_1d_low) * 1.1 / 6.0  # R1/S1 level
    camarilla_r1 = prev_1d_close + camarilla_width
    camarilla_s1 = prev_1d_close - camarilla_width
    
    # Align daily Camarilla levels to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # --- Volume confirmation (3x 20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 1d EMA34 (34 periods) and 20-period volume MA
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 3.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume surge and daily uptrend
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_surge and 
                ema_34_1d_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume surge and daily downtrend
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_surge and 
                  ema_34_1d_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S1 OR daily EMA34 turns down
                if (close[i] < camarilla_s1_aligned[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R1 OR daily EMA34 turns up
                if (close[i] > camarilla_r1_aligned[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals