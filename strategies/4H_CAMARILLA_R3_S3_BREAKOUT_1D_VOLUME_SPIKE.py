# 4H_CAMARILLA_R3_S3_BREAKOUT_1D_VOLUME_SPIKE
# Hypothesis: Camarilla R3/S3 levels on 1d timeframe act as strong support/resistance. Price breaking above R3 or below S3
# with volume > 2x 20-period average indicates institutional breakout. Trend filter: price > 1d EMA50 for longs, < EMA50 for shorts.
# Works in bull markets (breakouts above resistance in uptrend) and bear markets (breakdowns below support in downtrend).
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years).

name = "4H_CAMARILLA_R3_S3_BREAKOUT_1D_VOLUME_SPIKE"
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
    
    # Daily data for Camarilla calculation and filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla R3 and S3 levels
    # R3 = close + 1.1*(high-low)/2
    # S3 = close - 1.1*(high-low)/2
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # EMA50 for trend filter
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current 4h volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 2 * vol_ma
    
    # Align to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one day of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike in uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike in downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below R3 or trend reversal
            if (close[i] < camarilla_r3_aligned[i] or 
                close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above S3 or trend reversal
            if (close[i] > camarilla_s3_aligned[i] or 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals