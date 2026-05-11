#!/usr/bin/env python3
# 6h_1d_1w_Camarilla_Pivot_R3_S3_Breakout_Trend_Volume
# Hypothesis: Combining daily and weekly trend filters with 6h Camarilla R3/S3 breakouts
# and volume confirmation creates high-probability trend-following trades.
# In bull markets: weekly uptrend + daily uptrend + breakout above R3 + volume surge = long
# In bear markets: weekly downtrend + daily downtrend + breakdown below S3 + volume surge = short
# Weekly filter ensures we only trade with the dominant long-term trend, reducing false signals
# during counter-trend movements. Volume surge confirms institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_1d_1w_Camarilla_Pivot_R3_S3_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get multi-timeframe data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 6h Camarilla levels (R3, S3) from previous 6h bar ---
    # For 6h timeframe, we need to get 6h data from prices
    # Since we're on 6h timeframe, prices are already 6h bars
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # Set first values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_width = (prev_high - prev_low) * 1.1 / 2.0
    camarilla_r3 = prev_close + camarilla_width
    camarilla_s3 = prev_close - camarilla_width
    
    # --- 1d EMA50 for trend filter ---
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- 1w EMA200 for higher timeframe trend filter ---
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # --- Volume confirmation (2.0x 24-period average on 6h) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 1w EMA200 (200 weeks) and 24-period volume MA
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume surge, 1d EMA50 uptrend, and 1w EMA200 uptrend
            if (close[i] > camarilla_r3[i] and 
                volume_surge and 
                ema_50_1d_aligned[i] < close[i] and 
                ema_200_1w_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume surge, 1d EMA50 downtrend, and 1w EMA200 downtrend
            elif (close[i] < camarilla_s3[i] and 
                  volume_surge and 
                  ema_50_1d_aligned[i] > close[i] and 
                  ema_200_1w_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S3 OR 1d EMA50 turns down OR 1w EMA200 turns down
                if (close[i] < camarilla_s3[i] or 
                    close[i] < ema_50_1d_aligned[i] or 
                    close[i] < ema_200_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R3 OR 1d EMA50 turns up OR 1w EMA200 turns up
                if (close[i] > camarilla_r3[i] or 
                    close[i] > ema_50_1d_aligned[i] or 
                    close[i] > ema_200_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals