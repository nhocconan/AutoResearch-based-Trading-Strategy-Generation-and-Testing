#!/usr/bin/env python3
# 4h_12h_1d_Camarilla_R1_S1_Breakout_Trend_Volume
# Hypothesis: Combines 12h trend filter with 4-hour Camarilla R1/S1 breakouts and volume confirmation.
# In bull markets, 12h uptrend + 4h breakout above R1 captures early momentum.
# In bear markets, 12h downtrend + 4h breakdown below S1 captures early downside.
# Volume filter ensures breakouts have conviction, reducing false signals.
# Tight entry conditions target 20-40 trades/year to minimize fee drag.

name = "4h_12h_1d_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter and 1d data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h EMA34 for trend filter ---
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
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
    
    # --- Volume confirmation (1.5x 20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 12h EMA34 (34 periods) and 20-period volume MA
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume surge and 12h uptrend
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_surge and 
                ema_34_12h_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume surge and 12h downtrend
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_surge and 
                  ema_34_12h_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S1 OR 12h EMA34 turns down
                if (close[i] < camarilla_s1_aligned[i] or 
                    close[i] < ema_34_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R1 OR 12h EMA34 turns up
                if (close[i] > camarilla_r1_aligned[i] or 
                    close[i] > ema_34_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals