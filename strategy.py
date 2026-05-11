# 4h_12h1d_CamarillaBreakout_TrendVolume
# Hypothesis: Uses Camarilla pivot levels from daily timeframe for precise entry/exit, 
# combined with 12h EMA trend filter and volume confirmation. Enters long when price 
# crosses above S1 level in a 12h uptrend with above-average volume, short when price 
# crosses below R1 level in a 12h downtrend with above-average volume. 
# Works in both bull (buy breakouts above S1 in uptrend) and bear (sell breakdowns below R1 in downtrend).

name = "4h_12h1d_CamarillaBreakout_TrendVolume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivots and 12h data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_1d) < 2 or len(df_12h) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla Pivot Levels ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses current day's values
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculations
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # S1, S2, S3 (support levels)
    s1 = close_1d - (range_hl * 1.1 / 6)
    s2 = close_1d - (range_hl * 1.1 / 4)
    s3 = close_1d - (range_hl * 1.1 / 2)
    
    # R1, R2, R3 (resistance levels)
    r1 = close_1d + (range_hl * 1.1 / 6)
    r2 = close_1d + (range_hl * 1.1 / 4)
    r3 = close_1d + (range_hl * 1.1 / 2)
    
    # Align daily levels to 4h timeframe (use previous day's levels)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # --- 12h EMA20 for trend ---
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # --- Volume confirmation ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA (20) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h EMA direction
        uptrend = ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1] if i > 0 else ema_20_12h_aligned[i] > close[i]
        downtrend = ema_20_12h_aligned[i] < ema_20_12h_aligned[i-1] if i > 0 else ema_20_12h_aligned[i] < close[i]
        
        # Volume filter: above average volume
        vol_confirm = vol_ratio[i] > 1.2
        
        if position == 0:
            # Long: price breaks above S1 in uptrend with volume confirmation
            if uptrend and vol_confirm and close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R1 in downtrend with volume confirmation
            elif downtrend and vol_confirm and close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses below S2 or reverses below S1
                if close[i] < s2_aligned[i] or (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above R2 or reverses above R1
                if close[i] > r2_aligned[i] or (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals