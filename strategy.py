# 12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER
# Strategy: Camarilla pivot breakout on 12h with 1d trend filter and volume confirmation
# Rationale: Camarilla R3/S3 levels act as strong support/resistance. Breakouts with
# volume and 1d trend alignment capture momentum while avoiding counter-trend trades.
# Works in bull/bear markets via trend filter. Target: 15-30 trades/year on 12h.

name = "12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER"
timeframe = "12h"
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
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    camarilla_r3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low'])
    camarilla_s3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low'])
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Average volume for confirmation (20-period)
    avg_vol = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    avg_vol_aligned = align_htf_to_ltf(prices, df_1d, avg_vol)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(avg_vol_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with volume and 1d uptrend
            vol_condition = volume[i] > 1.5 * avg_vol_aligned[i]  # 50% above average
            if (close[i] > camarilla_r3_aligned[i] and 
                vol_condition and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume and 1d downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  vol_condition and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or trend reversal
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or trend reversal
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals