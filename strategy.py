# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_v2
# Hypothesis: Camarilla R1/S1 breakout with 12-hour trend filter and volume confirmation. 
# Uses price channel structure from daily pivots, trades in direction of higher timeframe trend,
# and filters with volume spikes to avoid false breakouts. Designed for 20-40 trades/year.
# Works in bull markets via breakouts and bear markets via avoidance of counter-trend trades.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla Pivot Levels (R1, S1) ---
    # Typical price: (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Pivot point
    pivot = typical_price.values
    # Range
    range_hl = df_1d['high'] - df_1d['low']
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = df_1d['close'].values + (range_hl * 1.1 / 12)
    s1 = df_1d['close'].values - (range_hl * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # --- 12h EMA50 for trend filter ---
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- Volume Spike Detection ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)  # Volume spike threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend based on 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Breakout signals
        long_breakout = (high[i] > r1_aligned[i]) and vol_spike[i]
        short_breakout = (low[i] < s1_aligned[i]) and vol_spike[i]
        
        if position == 0:
            # Only take long breakouts in uptrend, short breakouts in downtrend
            if uptrend and long_breakout:
                signals[i] = 0.25
                position = 1
            elif downtrend and short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to pivot level or opposite Camarilla level
            if position == 1:
                # Exit long: price touches S1 or returns below pivot
                exit_signal = (low[i] < s1_aligned[i]) or (close[i] < pivot[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches R1 or returns above pivot
                exit_signal = (high[i] > r1_aligned[i]) or (close[i] > pivot[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals