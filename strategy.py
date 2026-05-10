# 6h_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: For 6h timeframe, use daily Camarilla R3/S3 levels for breakout entries.
# In trending markets (12h EMA50), price breaks R3/S3 and continues; in ranging markets, fewer triggers.
# 12h trend filter avoids counter-trend trades. Volume confirmation reduces false breakouts.
# Designed for 6h to capture multi-day moves with low frequency (~15-30 trades/year).
# Works in bull (breakouts continue) and bear (breakdowns continue) via trend filter.

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_val = df_1d['high'] - df_1d['low']
    
    R3 = typical_price + (range_val * 1.1 / 2)
    S3 = typical_price - (range_val * 1.1 / 2)
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    R3_prev = R3.shift(1).values
    S3_prev = S3.shift(1).values
    
    # 12h trend: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align daily and 12h data to 6h
    R3_prev_aligned = align_htf_to_ltf(prices, df_1d, R3_prev)
    S3_prev_aligned = align_htf_to_ltf(prices, df_1d, S3_prev)
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # Volume confirmation: 4-period (1-day) average on 6h
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_prev_aligned[i]) or np.isnan(S3_prev_aligned[i]) or
            np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: break above R3 with 12h uptrend and volume
            if (close[i] > R3_prev_aligned[i] and 
                trend_12h_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: break below S3 with 12h downtrend and volume
            elif (close[i] < S3_prev_aligned[i] and 
                  trend_12h_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to typical price or trend fails
            typical_price_aligned = ((df_1d['high'] + df_1d['low'] + df_1d['close']) / 3).shift(1).values
            typical_price_aligned = align_htf_to_ltf(prices, df_1d, typical_price_aligned)
            if (close[i] < typical_price_aligned[i] or 
                trend_12h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to typical price or trend fails
            typical_price_aligned = ((df_1d['high'] + df_1d['low'] + df_1d['close']) / 3).shift(1).values
            typical_price_aligned = align_htf_to_ltf(prices, df_1d, typical_price_aligned)
            if (close[i] > typical_price_aligned[i] or 
                trend_12h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals