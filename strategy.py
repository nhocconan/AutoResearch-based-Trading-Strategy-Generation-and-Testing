# 4h_Camarilla_R3S3_12hTrendFilter_V1
# Hypothesis: 4h Camarilla Pivot R3/S3 Breakout with 12h EMA Trend Filter
# Enters long when price breaks above R3 with 12h EMA200 > 12h EMA50 (bullish trend) and volume > 1.5x average.
# Enters short when price breaks below S3 with 12h EMA200 < 12h EMA50 (bearish trend) and volume > 1.5x average.
# Exits when price returns to the pivot point (PP).
# Uses 12h EMA to filter for trend alignment, reducing false signals in ranging markets.
# Target: 20-50 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_12hTrendFilter_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    # Calculate 12h EMA50 and EMA200
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMA to 4h
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # === Camarilla Pivot Levels (from previous day) ===
    # Using previous day's OHLC to avoid lookahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Pivot Point
    pp = (prev_high + prev_low + prev_close) / 3.0
    
    # Camarilla levels
    r3 = pp + (prev_high - prev_low) * 1.1 / 4.0
    s3 = pp - (prev_high - prev_low) * 1.1 / 4.0
    
    # Align levels to 4h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === Volume confirmation (4h) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        pp_val = pp_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema50_val = ema50_12h_aligned[i]
        ema200_val = ema200_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(pp_val) or np.isnan(r3_val) or np.isnan(s3_val) or 
            np.isnan(ema50_val) or np.isnan(ema200_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3, 12h EMA200 > EMA50 (bullish trend), volume spike
            if close_val > r3_val and ema200_val > ema50_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3, 12h EMA200 < EMA50 (bearish trend), volume spike
            elif close_val < s3_val and ema200_val < ema50_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point
            if close_val <= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point
            if close_val >= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals