# 4H_Camarilla_R3_S3_Breakout_12HTF_Trend_Volume
# Hypothesis: Uses Camarilla R3/S3 levels from daily timeframe with 12-hour EMA trend filter and volume spike confirmation.
# The 12-hour EMA provides stronger trend filtering than daily EMA, reducing false signals during choppy periods.
# Only enters long on breakout above R3 when 12h EMA is rising (trend up) or short on breakdown below S3 when 12h EMA is falling (trend down).
# Exits when price returns inside the pivot range (S3 to R3) to capture mean reversion and avoid overtrading.
# Designed for low trade frequency (target 25-40 trades/year) and works in both bull and bear markets by following higher timeframe trend.

name = "4H_Camarilla_R3_S3_Breakout_12HTF_Trend_Volume"
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + rng * 1.1 / 4
    camarilla_s3 = close_1d - rng * 1.1 / 4
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12-hour EMA34 for trend filter (more responsive than daily)
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume filter: current volume > 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3 + Uptrend (12h EMA rising) + volume spike
            if (close[i] > r3_aligned[i] and 
                ema34_aligned[i] > ema34_aligned[i-1] and  # 12h EMA rising
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + Downtrend (12h EMA falling) + volume spike
            elif (close[i] < s3_aligned[i] and 
                  ema34_aligned[i] < ema34_aligned[i-1] and  # 12h EMA falling
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price returns inside pivot range (below R3 and above S3) - reversion to mean
            if close[i] < r3_aligned[i] and close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price returns inside pivot range (below R3 and above S3) - reversion to mean
            if close[i] < r3_aligned[i] and close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals