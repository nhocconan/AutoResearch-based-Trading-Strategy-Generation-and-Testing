#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot breakout on 12h with daily trend filter and volume confirmation.
- Long when: price breaks above R1, daily EMA34 uptrend, volume > 20-period average
- Short when: price breaks below S1, daily EMA34 downtrend, volume > 20-period average
- Exit when price returns to H4 (pivot mean) or trend reverses
Camarilla levels provide precise intraday support/resistance. Trend filter ensures
we trade with the higher timeframe momentum. Volume confirms breakout strength.
Targets 12-25 trades/year (48-100 over 4 years) to minimize fee drag.
Works in bull by catching breakouts, in bear by fading false breaks to S1/R1.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- Daily Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Camarilla Levels from Previous Day ---
    # Calculate from daily OHLC (previous day's values)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Camarilla multipliers
    R1 = close_1d_vals + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d_vals - (high_1d - low_1d) * 1.1 / 12
    
    # Align to 12h timeframe (values update only at daily close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Pivot (H4 level) for exit
    H4 = close_1d_vals + (high_1d - low_1d) * 1.1 / 2
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    
    # --- Volume Confirmation: 12h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(H4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend
        trend_up = close_12h[i] > ema34_1d_aligned[i]
        trend_down = close_12h[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_12h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for breakouts only in direction of daily trend with volume
            if close_12h[i] > R1_aligned[i] and trend_up and vol_ok:
                # Long: price breaks above R1 + daily uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close_12h[i] < S1_aligned[i] and trend_down and vol_ok:
                # Short: price breaks below S1 + daily downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to H4 (mean reversion) OR trend turns down
                if close_12h[i] <= H4_aligned[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to H4 OR trend turns up
                if close_12h[i] >= H4_aligned[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals