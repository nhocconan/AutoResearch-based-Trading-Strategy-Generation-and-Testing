#!/usr/bin/env python3
# 12h_1d_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Uses 1d Camarilla pivot levels (R3/S3) on 12h timeframe with 1d trend filter and volume confirmation.
# In bull markets: 1d uptrend + breakout above R3 captures momentum. In bear markets: 1d downtrend + breakdown below S3 captures short opportunities.
# Volume filter ensures breakouts have conviction, reducing false signals. Target: 12-37 trades/year to minimize fee drag.

name = "12h_1d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1-day data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla levels (R3, S3) from previous day ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_high = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_low = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Shift by 1 to use only completed 1d candle (avoid look-ahead)
    camarilla_high_shifted = np.roll(camarilla_high, 1)
    camarilla_low_shifted = np.roll(camarilla_low, 1)
    camarilla_high_shifted[0] = np.nan  # First value invalid after roll
    camarilla_low_shifted[0] = np.nan
    
    # Align 1d Camarilla levels to 12h
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high_shifted)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low_shifted)
    
    # --- 1d EMA34 for trend filter ---
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume confirmation (1.5x 20-period average on 12h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 1d EMA34 (34 periods) and 20-period volume MA
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_high_aligned[i]) or
            np.isnan(camarilla_low_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with volume surge and 1d uptrend
            if (close[i] > camarilla_high_aligned[i] and 
                volume_surge and 
                ema_34_1d_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 with volume surge and 1d downtrend
            elif (close[i] < camarilla_low_aligned[i] and 
                  volume_surge and 
                  ema_34_1d_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below Camarilla S3 OR 1d EMA34 turns down
                if (close[i] < camarilla_low_aligned[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above Camarilla R3 OR 1d EMA34 turns up
                if (close[i] > camarilla_high_aligned[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals